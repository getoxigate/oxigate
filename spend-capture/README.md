# spend-capture

Records the traffic between the gateway and its LLM providers, and plays it back to the gateway
later. The recordings are captured at the HTTP boundary, before any parsing, so they hold exactly
what the provider sent, not the gateway's interpretation of it. That makes them usable as ground
truth when checking what the gateway reports it spent.

- **`spend-capture capture`** is a recording proxy. Point a provider's `api_base_url` at it and
  it forwards every request to the real provider unchanged. Each response is relayed to the
  gateway as it arrives, and recorded once the exchange ends.
- **`spend-capture replay`** serves a capture session back to the gateway in order, with no
  provider involved. Every request must match the recording that is due next.

It is a test harness, not part of the gateway. The gateway needs no change and no special build
to use it: only configuration.

## Wiring the gateway

The harness listens for plain HTTP and forwards to the provider over TLS. Requests under
`/<route>` go to that route's upstream origin, so the gateway's base URL for a provider becomes
`http://<harness>/<route>`:

```yaml
providers:
  openai:
    api_base_url: "http://127.0.0.1:8480/openai"
  anthropic:
    api_base_url: "http://127.0.0.1:8480/anthropic"
  gemini:
    mode: api
    api_base_url: "http://127.0.0.1:8480/gemini"
```

```sh
spend-capture capture \
  --route openai=https://api.openai.com \
  --route anthropic=https://api.anthropic.com \
  --route gemini=https://generativelanguage.googleapis.com
```

A request for `/openai/v1/chat/completions` is forwarded to
`https://api.openai.com/v1/chat/completions`. Both subcommands listen on the same default
address, so one gateway configuration serves both the capture and its replay.

This works for any provider whose base URL can be set to an `http://` address. Azure OpenAI
requires `https://`, and Vertex AI's endpoint cannot be overridden, so neither can be captured
this way.

## Command line

### `capture`

| Option | Default | Meaning |
|---|---|---|
| `--route NAME=ORIGIN` | required, repeatable | Forward requests under `/NAME` to `ORIGIN`: a bare `scheme://host[:port]`, with no path, query or userinfo. Names are letters, digits, `-`, `_` and `.`. |
| `--listen ADDR` | `127.0.0.1:8480` | Where to accept the gateway's plain-HTTP requests. |
| `--results DIR` | the crate's `results/` | Results directory. It must not already hold recordings or incident records: one directory is one session. The default is the source tree's `spend-capture/results`, fixed at build time, not a path relative to where the binary runs. |
| `--declare-env VAR` | none, repeatable | Name of an environment variable holding a credential to scan bodies for, besides the ones each exchange carries (see [Credentials](#credentials)). The value is read from the environment so it never appears on the command line. |
| `--upstream-ca PEM` | none | Extra certificates to trust for upstream TLS, besides the bundled web roots. For a local test upstream. |

**Use `https://` origins for real providers.** An `http://` origin is accepted only because a
local mock upstream needs one. The harness never downgrades an `https://` origin.

Stop the capture with Ctrl-C. It stops taking new requests, lets any exchange in flight finish
and be written, then exits. A second Ctrl-C exits at once, and the exchanges still in flight
are not recorded.

### `replay`

| Option | Default | Meaning |
|---|---|---|
| `--listen ADDR` | `127.0.0.1:8480` | Where to accept the gateway's requests. |
| `--results DIR` | the crate's `results/` | The capture session to replay. It is only read. The default is fixed at build time, as for `capture`. |
| `--declare-env VAR` | none, repeatable | The same credentials declared to the capture, so they are scrubbed from everything the replay says about a request: its refusals and its log. Matching does not depend on them. |

Replay serves until Ctrl-C. It then waits for the requests still open to be answered, and exits
non-zero if a request was refused or any recording was never requested.

## What gets recorded

The capture forwards each request exactly as the gateway sent it. It adds no header, re-encodes
no body and follows no redirect. The one exception is transport headers (`Host`, `Connection`,
`Keep-Alive`, `TE`, `Trailer`, `Transfer-Encoding`, `Upgrade`, `Content-Length`, every `Proxy-*`
header, and any header named by `Connection`), which each hop's HTTP stack writes for itself.
Every other request and response header is recorded, including rate-limit, request-id and
deprecation headers, redacted only where it carries a credential.

An exchange is recorded only if it ended with a response. It is **not recorded**, only logged,
when:

- no response arrived: the upstream was unreachable or failed, or the gateway went away before
  the upstream answered;
- the upstream body failed part-way through.

A replay has to present an exchange as it actually happened, and the format has no way to
represent one of these endings.

An exchange that ended with a response is recorded either way it finished:

- **`upstream_complete`**: the provider's body ended.
- **`downstream_closed_first`**: the gateway stopped reading before the provider's body ended.
  This is normal for a successful stream, because the gateway stops at the provider's end-of-
  stream marker. The capture then closes the upstream connection rather than draining it, so it
  does not keep the provider generating. The recording holds exactly the bytes relayed to the gateway.

## Recording format (version 1)

```
results/
  capture.log                 harness log (redacted), one line per event
  recordings/
    00000000/                 one directory per recorded exchange, by sequence number
      meta.json
      request.body            request body, verbatim
      response.body           buffered response body, verbatim
                              (or response.sse for a text/event-stream response)
  incidents/
    00000003.json             exchanges refused because a body carried a credential
```

`meta.json`:

| Field | Meaning |
|---|---|
| `format_version` | `1`. Readers reject any other version. |
| `sequence` | Position of the exchange in the session, from 0. The capture assigns a number to every request as it arrives, so an exchange that was not recorded leaves a gap. |
| `route`, `upstream_origin` | The route the request arrived on, and where it was forwarded. |
| `method` | Request method. |
| `uri` | Path and query as sent upstream, after redaction. |
| `status` | Response status code. |
| `request_headers`, `response_headers` | Ordered `{name, value}` entries, lower-case names, transport headers left out, redacted. A value that is not UTF-8 is recorded as an array of bytes. |
| `request_body_file`, `response_body_file` | The body files' names within the directory. |
| `response_kind` | `buffered` or `event_stream`. |
| `response_chunks` | Sizes of the response body chunks, in the order the capture received them from upstream. They sum to the body file's length. |
| `termination` | `upstream_complete` or `downstream_closed_first`. |

An incident record holds `format_version`, `sequence`, `route`, `method`, `status` (`null` if no
response arrived), and `surfaces`: where credentials were found (`request_body`, `response_body`,
`response_stream`). It holds no body, header, URI or credential.

Recordings and incident records are committed atomically. Each is written in full under
`results/.staging/`, then renamed into place, so a reader never sees a partial one.

## Credentials

Redaction happens when a recording is written. What gets forwarded is never altered.

- **Headers** named `Authorization`, `Proxy-Authorization`, `x-api-key`, `api-key`,
  `x-goog-api-key`, `Cookie`, `Set-Cookie` or `X-Amz-Security-Token` (any case) are replaced with
  `[REDACTED]`. So is any header whose name has one of these as a whole segment: `key`, `apikey`,
  `token`, `secret`, `password`, `passwd`, `auth`, `authorization`, `session`, `cookie`,
  `credential`, `credentials`, `sig`, `signature`. That catches `X-Session-Token`, but not the
  providers' `...-tokens` rate-limit headers, which are kept.
- **URIs**: in the request URI and in `Location` / `Content-Location` values, query parameters
  with a credential name (such as `?key=`) and any userinfo are replaced with `[REDACTED]`.
- Any credential value collected from those carriers, plus any declared value, is also scrubbed
  wherever else it appears in a header or URI, and from the harness log.

**Bodies are never rewritten.** When a request body, response body or stream contains a credential
value, the harness forwards the exchange unchanged but records nothing for it: it writes an
incident record naming the exchange and where the value was found. A value split across two
stream chunks is still found, because the exchange is scanned as a whole before anything is
written. This fails closed. A very short credential that happens to appear in a body will refuse
that recording.

The values scanned for are those carried in the exchange's own credential headers and URI,
plus any declared with `--declare-env`. Declare the same values to the replay, so it never
repeats them.

Limits:

- The guarantee covers the files this harness writes: `results/` and its log on standard error.
  Other processes, including the gateway and the provider, are outside it.
- Only known values are found. A credential that appears in a body without also appearing in one
  of the exchange's carriers, and was not declared, cannot be recognised.
- Outside a carrier, redaction replaces the value wherever it occurs. A short credential that
  also occurs in, say, a path is redacted there too.
- A literal `[REDACTED]` sent by the gateway in a header or URI is recorded as a marker, so on
  replay it matches any value in that position. Telling the two apart would need an escape in
  the recording format, and the gateway sends no such value.

## Replay matching

A request matches the next recording when its route, method, redacted path and query, end-to-end
headers after redaction, and body bytes are all equal. Transport headers are ignored. Credentials
never cause a mismatch: each `[REDACTED]` in the recording stands for any value, and every byte
outside it must be equal. In the path and query, that value stays within one segment or parameter,
so a request that adds or drops a parameter does not match. A replay with a different API key or a different `Host` matches. So does
a request holding a value the capture redacted but the replay cannot recognise as a credential on
its own, such as one the response's `Set-Cookie` issued.

Nothing the replay writes, in a refusal or on standard error, repeats a credential the request
carries or one declared to the replay.

The replay refuses a request that differs, or one arriving after every recording has been served,
with status `409` and a message naming the field that differed. A request that matches a later
recording is reported as out of sequence. A body is described only by its length and the offset
of the first difference, never quoted. Where the recorded value holds `[REDACTED]`, the request's
differing value is not quoted either, since the difference may be a credential the replay was
not given.

Every request that reached the capture took the next sequence number, recorded or not. When the
position that is due has no recording (the exchange failed, was refused for a credential, or hit
an unknown route), the replay refuses that request as **not recorded** instead of serving the next
recording in its place. Serving it would answer a request with another exchange's response, such
as a failed attempt with its retry's success. The requests before the gap are served normally. A
corpus with gaps can therefore only be replayed up to its first gap. A missing exchange after the
last recording shows up as an unexpected request. After a refusal, the replay stops serving: every later
request is refused too, so a diverged session cannot serve a recording to the wrong request.

Requests are matched in the order they arrive, so drive the gateway one request at a time, in
the order of the capture.

A response is served with the recorded status, headers and body. The body is sent as the recorded
chunks, one at a time. HTTP and TCP may still merge chunks in transit, so the gateway can receive
fewer, larger ones. For a `downstream_closed_first` recording, the body fails after the last
recorded byte instead of ending. A gateway that stops at the provider's end-of-stream marker, as
it did during the capture, sees exactly what it saw then. One that reads further gets an error,
never a clean end to a response that never completed.

## Where run data goes

Recordings hold real prompts and completions. The default results directory,
`spend-capture/results/`, is ignored by this directory's `.gitignore` and by the repository's
`.dockerignore`, so it is never committed, never published with the source, and never sent into a
Docker build. If you choose another `--results` directory, you are responsible for keeping it
private.
