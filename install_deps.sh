#!/bin/bash
set -ex

BIN=`dirname ${0}`
BIN=`cd ${BIN}; pwd`

apt update --allow-insecure-repositories

apt -y install gcc make libtool autoconf  librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool  libibverbs-dev rdma-core strace --allow-unauthenticated
apt -y install openssh-server openmpi-bin openmpi-common libopenmpi-dev --allow-unauthenticated

echo -e "\n\n============Installing required pkgs============\n\n"
 
wget https://docs.broadcom.com/docs-and-downloads/ethernet-network-adapters/NXE/BRCM_233.1.135.7/bcm_233.1.135.7.tar.gz
tar xvzf bcm_233.1.135.7.tar.gz
cd bcm_233.1.135.7/drivers_linux/bnxt_rocelib/
echo -e "\n\n============Compiling RoCE Lib now============\n\n"
tar -xf libbnxt_re-233.0.152.2.tar.gz
     cd libbnxt_re-233.0.152.2
     sh autogen.sh
     ./configure
     make
     find /usr/lib64/  /usr/lib -name "libbnxt_re-rdmav*.so"  -exec mv {} {}.inbox \;
     make install all
     sudo sh -c "echo /usr/local/lib >> /etc/ld.so.conf"
     sudo ldconfig
     cp -f bnxt_re.driver /etc/libibverbs.d/
      
     find . -name "*.so" -exec md5sum {} \;
     BUILT_MD5SUM=$(find . -name "libbnxt_re-rdmav*.so" -exec md5sum {} \; |  cut -d " " -f 1)
     echo -e "\n\nmd5sum of the built libbnxt_re is $BUILT_MD5SUM"
cd ../../../../
echo $PWD
rm -rf bcm_233.1.135.7.tar.gz  bcm_233.1.135.7

echo "install glog"
TMP=${BIN}/tmp

rm -rf ${TMP} && mkdir -p ${TMP}
cd ${TMP}
git clone https://github.com/google/glog.git
cd glog
mkdir build
cd build/
cmake ..
make -j 10
make install

cd ${BIN}
rm -rf ${TMP}
