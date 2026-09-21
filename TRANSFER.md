# Transfer the source through Docker Hub

This is a data-only image. It does not contain Python, an operating system, or
an inference server. Do not use `docker run`; copy `/src` out of a stopped
container instead.

```bash
docker pull epsilon1234/langgraph-runtime-inmem-post4:0.33.3-post4-review-20260919
transfer_id="$(docker create epsilon1234/langgraph-runtime-inmem-post4:0.33.3-post4-review-20260919)"
docker cp "$transfer_id:/src" ./langgraph-runtime-inmem-post4
docker rm "$transfer_id"
```

Verify the bundled wheel after extraction:

```bash
sha256sum langgraph-runtime-inmem-post4/wheel/langgraph_runtime_inmem-0.33.3.post4+review.20260919-py3-none-any.whl
```

Expected SHA-256:

```text
d162e669188d7d2da53f7452948105c53f9fdc6a1a42d39ec3630476a1a98b58
```

The immutable version tag is preferred. `latest` points to the same source at
the time of publication but may move in a future release.

The image transfers this repository, the final wheel under `wheel/`, and the
review delta under `patches/`. It deliberately excludes Git history, virtual
environments, caches, `.env`, and `.langgraph_api` data. Installing the wheel
still requires its Python dependencies to be available on the destination
machine.
