# ===== PREP =====
mkdir external
pushd external
setup_root=$(pwd)

# For easier installation of perl modules
cpan -j perl/cpan/MyConfig.pm App::cpanminus

# ===== CAPR =====
git clone https://github.com/fukunagatsu/CapR.git
pushd CapR

git checkout 23769feb6a9b18c8b49b4a6d46ea4d2a1f978b0c
make

popd

# ===== BPRNA =====
./perl/bin/cpanm Graph --local-lib=./perl

git clone https://github.com/hendrixlab/bpRNA
pushd bpRNA

git checkout e775a612b72894b912b8cfa7c8ab191d86922b44

popd

# ===== CONTRAFOLD =====
git clone https://github.com/csfoo/contrafold-se
pushd contrafold-se/src

git checkout 91322024471d430681437c083a09cf1c9ce65e0e
sed -i "s/static const double DATA_LOW_THRESH = 1e-7;/static constexpr double DATA_LOW_THRESH = 1e-7;/g" InferenceEngine.hpp
make

popd

# ===== ETERNAFOLD =====
git clone https://github.com/eternagame/EternaFold.git
pushd EternaFold/src

git checkout f77216d82595040f64412b3920bdf4cb32f4a8a6
make

popd

# ===== VIENNA (FOR IPKNOT) =====
wget https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_6_x/ViennaRNA-2.6.4.tar.gz
tar -xvf ViennaRNA-2.6.4.tar.gz
rm ViennaRNA-2.6.4.tar.gz
pushd ViennaRNA-2.6.4

./configure --prefix=$setup_root/ViennaRNA-2.6.4
make
make check
make install
make installcheck

popd

# ===== GLPK (FOR IPKNOT) =====
wget https://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz
tar -xvf glpk-5.0.tar.gz
rm glpk-5.0.tar.gz
pushd glpk-5.0

./configure --prefix=$setup_root/glpk-5.0
make
make check
make install
popd

# ===== IPKNOT =====
git clone https://github.com/satoken/ipknot.git
pushd ipknot

git checkout aa1f2ac9569983c3a1027e0f20bdcfeba77fd4d4
export PKG_CONFIG_PATH=$setup_root/ViennaRNA-2.6.4/lib/pkgconfig:$PKG_CONFIG_PATH
mkdir build
pushd build
cmake -DCMAKE_BUILD_TYPE=Release ..

sed -i "s#GLPK_INCLUDE_DIR:PATH=.*#GLPK_INCLUDE_DIR:PATH=$setup_root/glpk-5.0/include#g" CMakeCache.txt
sed -i "s#GLPK_LIBRARY:FILEPATH=.*#GLPK_LIBRARY:FILEPATH=$setup_root/glpk-5.0/lib/libglpk.a#g" CMakeCache.txt
sed -i "s#GLPK_ROOT_DIR:PATH=.*#GLPK_ROOT_DIR:PATH=$setup_root/glpk-5.0#g" CMakeCache.txt
cmake --build .
cmake --install . --prefix $setup_root/ipknot

popd
popd

# ===== LINEARPARTITION =====
git clone https://github.com/LinearFold/LinearPartition
pushd LinearPartition

git checkout ae6507f3053573decd2e4bdae60d5a96eac87783
patch --ignore-whitespace -p1 < ../EternaFold/LinearPartition-E.patch
make

popd

# ===== ARNIEFILE ====
echo -e "vienna_2: $setup_root/ViennaRNA-2.6.4/src/bin" > ./arniefile.txt
echo -e "contrafold: $setup_root/contrafold-se/src" >> ./arniefile.txt
echo -e "eternafold: $setup_root/EternaFold/src" >> ./arniefile.txt
echo -e "ipknot: $setup_root/ipknot/bin" >> ./arniefile.txt
echo -e "linearpartition: $setup_root/LinearPartition/bin" >> ./arniefile.txt
echo -e "TMP: /tmp" >> ./arniefile.txt

# ===== NVIDIA APEX =====
git clone https://github.com/NVIDIA/apex
pushd apex

git checkout 48c4894c4b38b2b77cd7a0473ca665e89c9c148b
pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

popd

# ===== CLEANUP =====
popd
