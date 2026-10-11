// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! The plain-HTTP/1.1 accept loop shared by the capture and the replay.

use std::convert::Infallible;
use std::future::Future;
use std::sync::Arc;

use http::{Request, Response};
use http_body::Body;
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use hyper_util::server::graceful::GracefulShutdown;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio::task::JoinHandle;

/// Serves every connection accepted on a listener with one handler, until stopped.
pub(crate) struct Server {
    stop: watch::Sender<bool>,
    task: JoinHandle<()>,
}

impl Server {
    /// Starts accepting on `listener`. Each request is passed to `handler` with `state`; an
    /// accept error is passed to `on_accept_error` and the loop carries on.
    ///
    /// Each connection is set `TCP_NODELAY`, so a body frame is written when it is produced
    /// rather than held back to coalesce with the next one.
    pub(crate) fn spawn<T, H, F, B>(
        listener: TcpListener,
        state: Arc<T>,
        handler: H,
        on_accept_error: fn(&T, &std::io::Error),
    ) -> Self
    where
        T: Send + Sync + 'static,
        H: Fn(Arc<T>, Request<Incoming>) -> F + Copy + Send + Sync + 'static,
        F: Future<Output = Result<Response<B>, Infallible>> + Send + 'static,
        B: Body + Send + 'static,
        B::Data: Send,
        B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        let (stop, stop_rx) = watch::channel(false);
        let task = tokio::spawn(accept_loop(
            listener,
            state,
            handler,
            on_accept_error,
            stop_rx,
        ));
        Self { stop, task }
    }

    /// Stops accepting connections and stops reading new requests on open ones: idle
    /// connections close at once, busy ones after their current exchange. Does not wait.
    pub(crate) fn stop(&self) {
        let _ = self.stop.send(true);
    }

    /// Stops, and waits until every open connection has closed.
    pub(crate) async fn shutdown(&mut self) {
        self.stop();
        let _ = (&mut self.task).await;
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop();
    }
}

async fn accept_loop<T, H, F, B>(
    listener: TcpListener,
    state: Arc<T>,
    handler: H,
    on_accept_error: fn(&T, &std::io::Error),
    mut stop: watch::Receiver<bool>,
) where
    T: Send + Sync + 'static,
    H: Fn(Arc<T>, Request<Incoming>) -> F + Copy + Send + Sync + 'static,
    F: Future<Output = Result<Response<B>, Infallible>> + Send + 'static,
    B: Body + Send + 'static,
    B::Data: Send,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    let connections = GracefulShutdown::new();
    loop {
        let stream = tokio::select! {
            _ = stop.wait_for(|stopped| *stopped) => break,
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => stream,
                Err(err) => {
                    on_accept_error(&state, &err);
                    continue;
                }
            },
        };
        // Best effort: a socket that refuses the option still serves, only with coalesced writes.
        let _ = stream.set_nodelay(true);
        let state = Arc::clone(&state);
        let service = service_fn(move |request| handler(Arc::clone(&state), request));
        let connection = connections
            .watch(http1::Builder::new().serve_connection(TokioIo::new(stream), service));
        // A connection error ends only that connection; the handler accounts for each exchange.
        tokio::spawn(async move {
            let _ = connection.await;
        });
    }
    drop(listener);
    connections.shutdown().await;
}
