[shader("compute")]
[numthreads(1, 1, 1)]
void forward(uint3 id : SV_DispatchThreadID) {}

[shader("compute")]
[numthreads(1, 1, 1)]
void backward(uint3 id : SV_DispatchThreadID) {}
