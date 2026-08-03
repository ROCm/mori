#!/usr/bin/env bash
# Exactly how MORI-F1 was created, so it can be recreated identically. Its /root/mori_tdm and its
# JIT cache live in the container's own writable layer, not in a volume, so anything that recreates
# it has to go through docker commit or the tree and ~890s of cached builds go with it.
set +e
C=${C:-MORI-F1}
echo "== image =="
docker inspect -f '{{.Config.Image}} {{.Image}}' "$C"
echo "== devices =="
docker inspect -f '{{range .HostConfig.Devices}}{{.PathOnHost}}:{{.PathInContainer}}:{{.CgroupPermissions}}{{"\n"}}{{end}}' "$C"
echo "== device count =="
docker inspect -f '{{len .HostConfig.Devices}}' "$C"
echo "== binds =="
docker inspect -f '{{range .HostConfig.Binds}}{{.}}{{"\n"}}{{end}}' "$C"
echo "== mounts =="
docker inspect -f '{{range .Mounts}}{{.Type}} {{.Source}} -> {{.Destination}} rw={{.RW}}{{"\n"}}{{end}}' "$C"
echo "== flags =="
docker inspect -f 'net={{.HostConfig.NetworkMode}} ipc={{.HostConfig.IpcMode}} pid={{.HostConfig.PidMode}} priv={{.HostConfig.Privileged}} shm={{.HostConfig.ShmSize}}' "$C"
echo "== groups/caps/security =="
docker inspect -f 'groups={{.HostConfig.GroupAdd}} caps={{.HostConfig.CapAdd}} sec={{.HostConfig.SecurityOpt}}' "$C"
echo "== ulimits =="
docker inspect -f '{{range .HostConfig.Ulimits}}{{.Name}}={{.Soft}}:{{.Hard}} {{end}}' "$C"
echo "== env (filtered) =="
docker inspect -f '{{range .Config.Env}}{{.}}{{"\n"}}{{end}}' "$C" | grep -viE 'path=|ls_colors|token|key|secret' | head -20
echo "== entry/cmd/workdir =="
docker inspect -f 'entry={{.Config.Entrypoint}} cmd={{.Config.Cmd}} wd={{.Config.WorkingDir}} tty={{.Config.Tty}} oi={{.Config.OpenStdin}}' "$C"
echo "INSPECT_DONE"
