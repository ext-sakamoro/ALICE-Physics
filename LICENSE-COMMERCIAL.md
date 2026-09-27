# ALICE-Physics — Commercial License

Copyright (c) 2024-2026 Moroya Sakamoto. All rights reserved.

ALICE-Physics is dual-licensed:

- **AGPL-3.0-or-later** — see [LICENSE-AGPL](LICENSE-AGPL). Free to use,
  modify, and distribute under the AGPL v3 terms. AGPL is a strong
  copyleft: a game, application, firmware image, or service that links
  `alice-physics` — including through the C ABI / FFI, the Unity / UE5 /
  Godot bindings, the Python bindings, or the WebAssembly build — and is
  distributed or served to users must be released under the AGPL as well.

- **Commercial License** — a paid alternative that removes the AGPL
  copyleft obligations. Contact us (below) to obtain a Commercial License
  if any of the following applies to your use case:

  1. You want to integrate ALICE-Physics into a **closed-source product**
     — a shipped game, a simulation tool, a desktop or mobile
     application — without releasing your own source under AGPL-3.0.
  2. You want to embed ALICE-Physics into a **proprietary SaaS** or any
     service that provides its functionality to users over a network.
  3. You want to ship ALICE-Physics inside an **edge device, embedded
     system, or firmware image** distributed to customers or deployed
     as part of a product offering.
  4. You want to **redistribute** ALICE-Physics (verbatim or modified) as
     part of a proprietary library, engine, plugin, or bundle — including
     Unity / Unreal Engine marketplace packages and native plugins.
  5. Your organisation's policy, your publisher's requirements, or a
     console / platform NDA is **incompatible with AGPL-3.0** source
     disclosure.
  6. You need a **written indemnity**, a **warranty**, or **priority
     support** that the AGPL-3.0 (as an as-is public licence) does not
     provide.

## What the Commercial License grants

- **Unlimited use** of the current and all future releases of
  ALICE-Physics under proprietary / closed-source terms.
- **Redistribution** of derivative works and modifications without the
  AGPL copyleft requirement, including in binary-only form.
- **No source-code disclosure** obligations for your own code that
  incorporates ALICE-Physics.
- **Determinism artefacts for verification** — access to the
  cross-platform bit-exactness golden vectors (`tests/test_det_golden.rs`
  fixtures, the Fix128 hi/lo FFI contract) so your build pipeline can
  assert bit-pattern equality across the platforms you ship on.
- **Written licence agreement** enumerating scope, term, and any
  indemnity / support terms negotiated.

## What the Commercial License does NOT grant

- Rights to **third-party dependencies**. ALICE-Physics depends on
  crates that keep their own licences (for example `alice-det-math`,
  `MIT OR Apache-2.0`). Those terms continue to apply independently of
  this licence. Optional feature bridges may pull additional
  AGPL-licensed ALICE crates — see the feature table in
  [`README.md`](README.md).
- Trademark rights over the "ALICE" or "ALICE-Physics" names — see
  [TRADEMARK_NOTICE](TRADEMARK_NOTICE).

## How to obtain a Commercial License

Contact:

- **Commercial licence enquiries** — `contact@extoria.co.jp`

Please describe:

1. Your organisation and product.
2. The scope of use (shipped game / SaaS / edge device / firmware /
   library redistribution / engine plugin / etc.).
3. Target platforms and expected deployment scale (units, seats, or
   monthly active users).
4. Any specific terms you need (indemnity, priority support,
   determinism certification, etc.).

A quotation will be provided based on the scope of use. Licences are
typically issued as an annual or perpetual grant.

## Relationship to the AGPL option

Choosing the AGPL-3.0-or-later option costs nothing and carries no
reporting obligation to us. The Commercial License exists for the cases
listed above, where AGPL source disclosure is not something you are able
or willing to do. You do not need to tell us which option you use unless
you are taking the Commercial License.

## No warranty

The AGPL-3.0 licence provides ALICE-Physics on an as-is basis with no
warranty. The Commercial License terms may include a limited warranty
negotiated in the written agreement.
