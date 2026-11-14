# ainic-netmon container image: the collector binary plus the Python report tool.
#
# Build (from the repository root):
#   docker build -f tools/netmon/docker/Dockerfile -t ainic-netmon tools/netmon
# Record (privileged: all sources; --network host is required for NIC discovery and ethtool):
#   docker run --rm --privileged --network host -v /shared/run1:/shared/run1 ainic-netmon \
#     record -o /shared/run1/{host} -d 60s --tag role=D
# Report:
#   docker run --rm -v /shared/run1:/shared/run1 --entrypoint ainic-netmon-report ainic-netmon \
#     report /shared/run1/nodeA /shared/run1/nodeB -o /shared/run1/report.html

# Build on Ubuntu 22.04 so the binary runs on any glibc >= 2.35 host or container.
FROM ubuntu:22.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ cmake make \
    && rm -rf /var/lib/apt/lists/*
COPY . /src
RUN cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release -DNETMON_BUILD_TESTS=ON \
    && cmake --build /build -j"$(nproc)" \
    && /build/tests/netmon_tests

FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
    && pip3 install --no-cache-dir numpy plotly \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /build/ainic-netmon /usr/local/bin/ainic-netmon
COPY report/ainic_netmon_report.py /usr/local/bin/ainic-netmon-report
RUN chmod 0755 /usr/local/bin/ainic-netmon-report
ENTRYPOINT ["ainic-netmon"]
