FROM scratch

LABEL org.opencontainers.image.source="https://github.com/Duy1230/langgraph-runtime-inmem-post4" \
      org.opencontainers.image.version="0.33.3.post4+review.20260919" \
      org.opencontainers.image.licenses="Elastic-2.0" \
      org.opencontainers.image.description="Source-transfer image; extract /src instead of running it"

COPY . /src/
WORKDIR /src

# This image intentionally has no runtime. Use docker create + docker cp.
CMD ["/transfer-only"]
