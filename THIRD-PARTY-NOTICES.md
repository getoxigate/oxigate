# Third-Party Notices

OxiGate itself is licensed under the GNU Affero General Public License v3.0 or later; see
[`LICENSE`](LICENSE). This file reproduces the notices required by the licences of third-party
material redistributed inside this repository.

If you redistribute OxiGate, in source or binary form, this file must travel with it. The bundled
pricing data is compiled into the binary by `include_bytes!`, so a compiled OxiGate contains the
material below whether or not the JSON file ships alongside it.

---

## Bundled model pricing data

**Component:** `assets/model_prices.json` (embedded into the binary at build time by
`src/domain/pricing.rs`)

Part of this file is derived from a third-party, community-maintained model-pricing dataset made
available under the MIT License. Entries carry a `source` of `aggregated-dataset` where they were
imported from it; the exact upstream revision each import was taken from is recorded in the file's
own `_provenance.revision` field.

The dataset is redistributed here under the terms below. The material imported comes from the
upstream repository root, outside the `enterprise/` directory that the first paragraph carves out,
so the MIT terms apply to all of it.

```
Portions of this software are licensed as follows:

* All content that resides under the "enterprise/" directory of this repository, if that
  directory exists, is licensed under the license defined in "enterprise/LICENSE".
* Content outside of the above mentioned directories or restrictions above is available under
  the MIT license as defined below.
---
MIT License

Copyright (c) 2023 Berri AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The copyright line above is reproduced exactly as it appears upstream. Do not expand, abbreviate,
or re-style it — MIT conditions redistribution on retaining the notice as given, and "Berri AI"
is what it says.

---

## Adding to this file

Any future import of third-party material under a licence with an attribution or notice condition
gets an entry here: what was imported, where it lives in this tree, and the licence text verbatim.
A copyright line on its own does not satisfy MIT — the permission notice and warranty disclaimer
are part of what the licence requires you to carry.
