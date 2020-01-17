
#include "hot_zone_kill.h"
#include <time.h>
#define REPEATS 5000
#define N 10


void visualize(struct Ball balls[], double t0, double tick, double delta_t, int width, int height, int a, int b, int top_border, int bottom_border, int right_border, int left_border, double R, int n, double l, double mu){
	return;
}

void kernel(struct Params *params, int *result){
	bool running = true;
	double tick = 0.0;
	int n_count = 0;
	int d_count = 0;
	int frame_counter = 0;
	int tick_counter = 0;
	int write_counter = 0;
	enum State state = GAME_ON;
	struct Ball balls[params->n];

	balls_init(balls, params);
	
	while (running) {
		tick_counter++;
		int putting_out_id = -1;
		tick += shortcut_step(balls, params, tick, &state, &putting_out_id) * params->delta_t;
		if(params->motion_mode == UNIFORMLY_DECELERATED && putting_out_id != -1){
			tick += ud_putting_out(balls, params, putting_out_id, tick);
			continue;
		}
		if(isnan(tick)) state = check_table(balls, params, tick);
		if(state == BALL_BEYOND_TABLE && running){
			running = false;
			(*result)++;
			printf("1\n");
		}
		if(state == LACK_OF_ENERGY && running){
			running = false;
			printf("0\n");
		}
		if(running){
			mechanics_step(balls, params, tick, &n_count, &d_count);
			tick += params->delta_t;
		}
	}
}

int main() {
	clock_t start, end;
    double cpu_time_used; 
    start = clock();

	srand(2137);

	struct Params *params = malloc(sizeof(struct Params));
	int c = 0;
	int *results = &c;
	params->width = 2*640;
	params->height = 2*480;
	params->a = 2*600;
	params->b = 2*400;
	params->R = 8.0;
	params->top_border = (params->height - params->b) / 2;
	params->bottom_border = (params->height + params->b) / 2;
	params->right_border = (params->width + params->a) / 2;
	params->left_border = (params->width - params->a) / 2;
	params->mu = 0.0002;
	params->n = N;
	params->l = 60.0;
	params->k = 1;
	params->delta_t = 0.001;
	params->v_max = 0.4*sqrt(2);
	params->motion_mode = PROPORTIONAL_TO_VELOCITY;
	//params->motion_mode = UNIFORMLY_DECELERATED;

	for(int i=0; i<REPEATS; i++){
		kernel(params, results);
	}
	end = clock();
	printf("p(c)=%lf\n", (double)c / REPEATS);
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf( "Time to generate:  %f s\n", cpu_time_used);
	//balls_init_from_file(balls, n, R, width, height, a, b, v_max);
	return 0;
}

