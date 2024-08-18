pushd operon
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_CLI_PROGRAMS=OFF \
    -DCMAKE_CXX_FLAGS="-march=x86-64 -mavx2 -mfma" \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
    -DCMAKE_PREFIX_PATH=${CONDA_PREFIX}/lib64/cmake
cmake --build build -j -t operon_operon -- VERBOSE=1
cmake --install build
popd operon
pushd pyoperon
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-march=x86-64 -mavx2 -mfma" \
    -DCMAKE_INSTALL_PREFIX=${PYTHON_SITE} \
    -DCMAKE_PREFIX_PATH=${CONDA_PREFIX}/lib64/cmake
cmake --build build -j -t pyoperon_pyoperon -- VERBOSE=1
cmake --install build
popd pyoperon