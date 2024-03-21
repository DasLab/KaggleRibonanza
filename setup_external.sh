set -e

# ===== PREP =====
echo "[KaggleRibonanza] Preparing..."

mkdir -p external
pushd external
setup_root=$(pwd)

# For easier installation of perl modules
if command -v cpanm &> /dev/null; then
    CPANM="cpanm"
else
    cpan -j perl/cpan/MyConfig.pm App::cpanminus
    CPANM="./perl/bin/cpanm"
fi

# ===== CAPR =====
echo "[KaggleRibonanza] Setting up CapR..."

if [ ! -d CapR ]; then
    git clone https://github.com/fukunagatsu/CapR.git
    pushd CapR
else
    pushd CapR
    git fetch
fi

git checkout 23769feb6a9b18c8b49b4a6d46ea4d2a1f978b0c
make

popd

# ===== BPRNA =====
echo "[KaggleRibonanza] Setting up bpRNA..."

$CPANM Graph --local-lib=./perl

if [ ! -d bpRNA ]; then
    git clone https://github.com/hendrixlab/bpRNA
    pushd bpRNA
else
    pushd bpRNA
    git fetch
fi

git checkout e775a612b72894b912b8cfa7c8ab191d86922b44

popd

# ===== CONTRAFOLD =====
echo "[KaggleRibonanza] Setting up Contrafold..."

if [ ! -d contrafold-se ]; then
    git clone https://github.com/csfoo/contrafold-se
    pushd contrafold-se/src
else
    pushd contrafold-se/src
    git fetch
fi

git checkout 91322024471d430681437c083a09cf1c9ce65e0e
sed -i "s/static const double DATA_LOW_THRESH = 1e-7;/static constexpr double DATA_LOW_THRESH = 1e-7;/g" InferenceEngine.hpp
make

popd

# ===== ETERNAFOLD =====
echo "[KaggleRibonanza] Setting up Eternafold..."

if [ ! -d EternaFold ]; then
    git clone https://github.com/eternagame/EternaFold.git
    pushd EternaFold/src
else
    pushd EternaFold/src
    git fetch
fi

git checkout f77216d82595040f64412b3920bdf4cb32f4a8a6
make

popd

# ===== VIENNA (FOR IPKNOT) =====
echo "[KaggleRibonanza] Setting up Vienna..."

if [ ! -d ViennaRNA-2.6.4 ]; then
    wget https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_6_x/ViennaRNA-2.6.4.tar.gz
    tar -xvf ViennaRNA-2.6.4.tar.gz
    rm ViennaRNA-2.6.4.tar.gz
fi
pushd ViennaRNA-2.6.4

./configure --prefix=$setup_root/ViennaRNA-2.6.4 --without-perl --without-python --without-forester --without-kinfold --without-rnalocmin --without-rnaxplorer
make
make check
make install
make installcheck

popd

# ===== GLPK (FOR IPKNOT) =====
echo "[KaggleRibonanza] Setting up GLPK..."

if [ ! -d glpk-5.0 ]; then
    wget https://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz
    tar -xvf glpk-5.0.tar.gz
    rm glpk-5.0.tar.gz
fi
pushd glpk-5.0

./configure --prefix=$setup_root/glpk-5.0
make
make check
make install
popd

# ===== IPKNOT =====
echo "[KaggleRibonanza] Setting up ipknot..."

if [ ! -d ipknot ]; then
    git clone https://github.com/satoken/ipknot.git
    pushd ipknot
else
    pushd ipknot
    git fetch
fi

git checkout aa1f2ac9569983c3a1027e0f20bdcfeba77fd4d4
export PKG_CONFIG_PATH=$setup_root/ViennaRNA-2.6.4/lib/pkgconfig:$PKG_CONFIG_PATH
mkdir -p build
pushd build
cmake -DCMAKE_BUILD_TYPE=Release -DGLPK_INCLUDE_DIR:PATH=$setup_root/glpk-5.0/include -DGLPK_LIBRARY:FILEPATH=$setup_root/glpk-5.0/lib/libglpk.a -DGLPK_ROOT_DIR:PATH=$setup_root/glpk-5.0 ..

cmake --build .
cmake --install . --prefix $setup_root/ipknot

popd
popd

# ===== LINEARPARTITION =====
echo "[KaggleRibonanza] Setting up LinearPartition..."

if [ ! -d LinearPartition ]; then
    git clone https://github.com/LinearFold/LinearPartition
    pushd LinearPartition
else
    pushd LinearPartition
    git fetch
fi

git checkout ae6507f3053573decd2e4bdae60d5a96eac87783
# In order to make sure our patch applies cleanly, throw out any prior modifications
# We use git stash instead of git clean just in case there were intentional local modifications
# someone wants to retrieve (the overhead incurred here should not be substantial)
git stash -u
patch --ignore-whitespace -p1 < ../EternaFold/LinearPartition-E.patch
make

popd

# ===== ARNIEFILE ====
echo "[KaggleRibonanza] Setting up arniefile..."

echo -e "vienna_2: $setup_root/ViennaRNA-2.6.4/src/bin" > ./arniefile.txt
echo -e "contrafold: $setup_root/contrafold-se/src" >> ./arniefile.txt
echo -e "eternafold: $setup_root/EternaFold/src" >> ./arniefile.txt
echo -e "ipknot: $setup_root/ipknot/bin" >> ./arniefile.txt
echo -e "linearpartition: $setup_root/LinearPartition/bin" >> ./arniefile.txt
echo -e "TMP: /tmp" >> ./arniefile.txt

# ===== NVIDIA APEX =====
echo "[KaggleRibonanza] Setting up Apex..."

if [ ! -d apex ]; then
    git clone https://github.com/NVIDIA/apex
    pushd apex
else
    pushd apex
    git fetch
fi

git checkout 48c4894c4b38b2b77cd7a0473ca665e89c9c148b
pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

popd

# ===== CLEANUP =====
echo "[KaggleRibonanza] Cleaning up..."
popd
