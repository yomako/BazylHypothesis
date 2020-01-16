#include "book.h"
#include "hot_zone_kill_gpu.h"
#define REPEATS 5000
#define N 10


struct BallsSet
{
	Ball balls[REPEATS][N];
};

__constant__ Params params;

__global__ void kernel(BallsSet *balls_from_host, int *results){
	__shared__ Ball balls[N];
	__shared__ bool running;
	
	int ball_id = threadIdx.x;
	enum State state = GAME_ON;
	double tick = 0.0;
	running = true;

	balls[ball_id].x = balls_from_host->balls[blockIdx.x][ball_id].x;
	balls[ball_id].y = balls_from_host->balls[blockIdx.x][ball_id].y;
	balls[ball_id].v_x = balls_from_host->balls[blockIdx.x][ball_id].v_x;
	balls[ball_id].v_y = balls_from_host->balls[blockIdx.x][ball_id].v_y;
	balls[ball_id].tick_base = balls_from_host->balls[blockIdx.x][ball_id].tick_base;
	__syncthreads();

	while (running) {
		__shared__ double lowests[N];
		__shared__ double lowest;
		__shared__ bool pocket_is_lowest;
		__shared__ bool pockets[N];
		shortcut_step_part1(balls, &params, ball_id, tick, &state, lowests, pockets);
		if(ball_id == 0){
			lowest = nan("1");
			state = GAME_ON;
			pocket_is_lowest = false;
			int lowest_id = 0;
			for(int i = 0; i < N; i++){
				if(is_lower(lowests[i], lowest)){
					lowest = lowests[i];
					lowest_id = i;
				}
			}
			pocket_is_lowest = pockets[lowest_id];
		}
		__syncthreads();
		tick += shortcut_step_part2(balls, &params, ball_id, lowest, pocket_is_lowest, tick, &state) * params.delta_t;
		if(ball_id == 0){
			if(isnan(tick)) state = check_table(balls, &params, tick);
			if(state == BALL_BEYOND_TABLE && running){
				running = false;
				(*results)++;
			}
			if(state == LACK_OF_ENERGY && running){
				running = false;
			}
		}
		__syncthreads();
		if(running){
			mechanics_step(balls, &params, ball_id, tick);
			tick += params.delta_t;
		}
		__syncthreads();
	}
}

int main( void ) {
	cudaEvent_t start, stop;
	HANDLE_ERROR( cudaEventCreate( &start ) );
	HANDLE_ERROR( cudaEventCreate( &stop ) );
	HANDLE_ERROR( cudaEventRecord( start, 0 ) );
	srand(2137); 
	bool running = true;
	Params pParams;
	pParams.width = 2*640;
	pParams.height = 2*480;
	pParams.a = 2*600;
	pParams.b = 2*400;
	pParams.R = 8.0;
	pParams.top_border = (pParams.height - pParams.b) / 2;
	pParams.bottom_border = (pParams.height + pParams.b) / 2;
	pParams.right_border = (pParams.width + pParams.a) / 2;
	pParams.left_border = (pParams.width - pParams.a) / 2;
	pParams.mu = 0.0002;
	pParams.l = 60.0;
	pParams.k = 1;
	pParams.delta_t = 0.001;
	pParams.v_max = 0.4*sqrt(2);
	pParams.n = N;

	Ball balls[REPEATS][N];
	BallsSet pBallsset;
	BallsSet *ballsset;
	int pResults = 0;
	int *results;

	HANDLE_ERROR(cudaMemcpyToSymbol(params, &pParams, sizeof(Params)));

	for(int i=0; i < REPEATS; i++){
		balls_init(balls[i], &pParams);
		
	}
	for(int i=0; i < REPEATS; i++){
		memcpy(pBallsset.balls[i], balls[i], N * sizeof (Ball));
	}
	printf("size=%lu, %lu\n", sizeof(Ball), SIZE_MAX);
	HANDLE_ERROR( cudaMalloc( (void**)&ballsset, sizeof(BallsSet) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&results, sizeof(int) ) );
	HANDLE_ERROR( cudaMemcpy( ballsset, &pBallsset, sizeof(BallsSet), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( results, &pResults, sizeof(int), cudaMemcpyHostToDevice ) );
	
	kernel<<<REPEATS,N>>>(ballsset, results);

	HANDLE_ERROR( cudaMemcpy( &pResults, results, sizeof(int), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	float   elapsedTime;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
	printf("p(c)=%lf\n", (double) pResults / REPEATS);
	printf( "Time to generate:  %f s\n", elapsedTime / 1000);
	
	HANDLE_ERROR( cudaFree( ballsset ) );
	HANDLE_ERROR( cudaFree( results ) );
	return 0;
}