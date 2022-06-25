||128|256|512|1024|
|-|-|-|-|-|
|OpenACC|0m0.549s|0m1.602s|0m6.232s|0m49.798s|
|cuBLAS|0m0.937s|0m2.013s|0m7.824s|1m4.385s|
|CUDA+cub|0m0.256s|0m0.569s|0m3.042s|0m35.336s|

||cudaMalloc|cudaLaunchKernel|cudaMemcpy|cudaFree|iteration|DeviceReduceKernel|maxKernel|DeviceReduceSingleTileKernel|
|-|-|-|-|-|-|-|-|-|
|Before opimization|195,29ms|105,56ms|32,19ms|0,028ms|110,18ms|1,22ms|1,05ms|1,03ms|
|After optimization|192,77ms|60,15ms|51,29ms|0,019ms|94,88ms|0,76ms|0,73ms|0,64ms|

||128|256|512|
|-|-|-|-|
|8x8|134445020|n/a|n/a|
|16x16|114034084|779651664|n/a|
|32x32|140437959|532797940|n/a|
|64x64|283168469|890902028|3900380632|
|128x128|796168689|2663046020|8737436531|
