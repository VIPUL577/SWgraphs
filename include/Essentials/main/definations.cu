// using distanceArray = unsigned long long int; 
// __device__ __host__
// int getDistance(distanceArray word){
//     return (static_cast<int>(word>>32)) ; 
// }
// __device__ __host__
// int getParent(distanceArray word){
//     return static_cast<int>(word) & 0xFFFFFFFF; 
// }
// __device__ __host__
// distanceArray getWord(int distance, int parent){
//    return (static_cast<distanceArray>(distance)<<32) | static_cast<distanceArray>(parent); 
// }

using distanceArray = unsigned long long int;

__device__ __host__
int getDistance(distanceArray word) {
    return static_cast<int>(word >> 32);
}

__device__ __host__
int getParent(distanceArray word) {
    return static_cast<int>(static_cast<uint32_t>(word));
}

__device__ __host__
distanceArray getWord(int distance, int parent) {
    return (static_cast<distanceArray>(static_cast<uint32_t>(distance)) << 32)
           | static_cast<uint32_t>(parent);
}