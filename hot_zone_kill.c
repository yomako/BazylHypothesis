
#include "hot_zone_kill.h"
#include <time.h>


void visualize(struct Ball balls[], double t0, double tick, double delta_t, int width, int height, int a, int b, int top_border, int bottom_border, int right_border, int left_border, double R, int n, double l, double mu){
	return;
}


int main() {
	srand(time(NULL)); 
	bool running = true;

	int width = 2*640, height = 2*480;
	int a = 2*400, b = 2*400;
	double R = 8.0;
	int top_border = (height - b) / 2;
	int bottom_border = (height + b) / 2;
	int right_border = (width + a) / 2;
	int left_border = (width - a) / 2;
	double mu = 0.0002;
	//double mu = 0.0;
	int n = 20;
	double l = 0.0;
	double k = 1;
	struct Ball balls[n];
	double tick = 0.0;
	double delta_t = 0.001;
	double v_max = 0.4*sqrt(2);
	int n_count = 0;
	int d_count = 0;
	int frame_counter = 0;
	int tick_counter = 0;
	bool shortcut_state = true;
	int write_counter = 0;
	FILE * fp;
	fp = fopen ("distribution.txt","w");
	
	balls_init(balls, n, R, width, height, a, b, v_max);
	//balls_init_from_file(balls, n, R, width, height, a, b, v_max);
	
	while (running) {
		tick_counter++;
		if(running){
			enum State state = GAME_ON;
			tick += shortcut_step(balls, top_border, bottom_border, left_border, right_border, R, mu, tick, delta_t, n, k, l, &state, width, height, a, b) * delta_t;
			if(isnan(tick)) state = check_table(balls, n, top_border, bottom_border, left_border, right_border, v_max, tick, mu, l);
			if(state == BALL_BEYOND_TABLE && running){
				printf("BALL_BEYOND_TABLE\n");
				running = false;
			}
			if(state == LACK_OF_ENERGY && running){
				printf("LACK_OF_ENERGY\n");
				printf("n_c=%d\n", n_count);
				running = false;
			}
			mechanics_step(balls, top_border, bottom_border, left_border, right_border, R, mu, tick, delta_t, n, k, l, &n_count, &d_count);
			printf("tick=%d, move=%d\n", tick_counter, all_balls_in_move(balls, n));
			if(all_balls_in_move(balls, n)) write_velocity_distribution(fp, balls, n, &write_counter);
		}
		
		tick += delta_t;
		//printf("\n");
	}

	fclose (fp);

	return 0;
}

