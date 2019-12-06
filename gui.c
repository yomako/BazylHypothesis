#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>
#include <allegro5/allegro_image.h>
#include "hot_zone_kill.h"


void draw_table(struct Ball balls[], int width, int height, int a, int b, int top_border, int bottom_border, int right_border, int left_border, double R, int n, double l){
	al_flip_display();
	al_draw_filled_rectangle(0, 0, width, height, al_map_rgb(0, 0, 0));
	int thickness = 5;
	al_draw_rectangle(left_border, top_border, right_border, bottom_border, al_map_rgb(255, 0, 0), thickness);
	al_draw_line(left_border, top_border, left_border + l/sqrt(2), top_border, al_map_rgb(0, 0, 0), thickness);
	al_draw_line(left_border, top_border, left_border-2, top_border + l/sqrt(2), al_map_rgb(0, 0, 0), thickness);
	al_draw_line(left_border, bottom_border, left_border + l/sqrt(2), bottom_border, al_map_rgb(0, 0, 0), thickness);
	al_draw_line(left_border, bottom_border - l/sqrt(2), left_border, bottom_border+1, al_map_rgb(0, 0, 0), thickness);
	al_draw_line(right_border, top_border, right_border - l/sqrt(2), top_border, al_map_rgb(0, 0, 0), thickness);
	al_draw_line(right_border, top_border, right_border, top_border + l/sqrt(2), al_map_rgb(0, 0, 0), thickness);
	al_draw_line(right_border, bottom_border, right_border - l/sqrt(2), bottom_border, al_map_rgb(0, 0, 0), thickness);
	al_draw_line(right_border, bottom_border, right_border, bottom_border - l/sqrt(2), al_map_rgb(0, 0, 0), thickness);

	//al_draw_line(left_border, -left_border+top_border+(left_border + l/sqrt(2)), left_border+l/sqrt(2), -left_border-l/sqrt(2)+top_border+(left_border + l/sqrt(2)), al_map_rgb(0, 255, 0), 1);
	//al_draw_line(left_border, left_border+bottom_border-(left_border + l/sqrt(2)), left_border+l/sqrt(2), left_border+l/sqrt(2)+bottom_border-(left_border + l/sqrt(2)), al_map_rgb(0, 255, 0), 1);
	if(width > height){
		al_draw_line((width-l)/2, top_border, (width+l)/2, top_border, al_map_rgb(0, 0, 0), thickness);
		al_draw_line((width-l)/2, bottom_border, (width+l)/2, bottom_border, al_map_rgb(0, 0, 0), thickness);
	}
	for (int i = 0; i < n; ++i)
	{
		if(!balls[i].color.filled) al_draw_ellipse(round(balls[i].x), round(balls[i].y), R, R, al_map_rgb(balls[i].color.r, balls[i].color.g, balls[i].color.b), 1);
		else al_draw_filled_ellipse(round(balls[i].x), round(balls[i].y), R, R, al_map_rgb(balls[i].color.r, balls[i].color.g, balls[i].color.b));
	}
}

void visualize(struct Ball balls[], double t0, double tick, double delta_t, int width, int height, int a, int b, int top_border, int bottom_border, int right_border, int left_border, double R, int n, double l, double mu){
	int mi = 1;
	while(t0+mi*delta_t < tick){
		double dx[n];
		double dy[n];
		for(int i=0; i<n; i++){
			dx[i] = movement_integral(balls[i].v_x, mu, t0+mi*delta_t-balls[i].tick_base, t0-balls[i].tick_base);
			dy[i] = movement_integral(balls[i].v_y, mu, t0+mi*delta_t-balls[i].tick_base, t0-balls[i].tick_base);
			balls[i].x += dx[i];
			balls[i].y += dy[i];
			//if(i==0) printf("x=%lf, y=%lf\n", balls[i].x, balls[i].y);
		}
		draw_table(balls, width, height, a, b, top_border, bottom_border, right_border, left_border, R, n, l);
		for(int i=0; i<n; i++){
			balls[i].x -= dx[i];
			balls[i].y -= dy[i];
		}
		mi++;
	}
}

int main() {
	ALLEGRO_DISPLAY * display;
	srand(time(NULL)); 
	al_init();
	al_init_image_addon();
	al_init_primitives_addon();
	bool running = true;

	int width = 2*640, height = 2*480;
	display = al_create_display(width, height);
	int a = 2*600, b = 2*400;
	double R = 10.0;
	int top_border = (height - b) / 2;
	int bottom_border = (height + b) / 2;
	int right_border = (width + a) / 2;
	int left_border = (width - a) / 2;
	double mu = 0.0004;
	//double mu = 0.0;
	int n = 15;
	double l = 60.0;
	double k = 1;
	struct Ball balls[n];
	double tick = 0.0;
	double delta_t = 0.001;
	double v_max = 4*sqrt(2);
	int n_count = 0;
	int d_count = 0;
	int frame_counter = 0;
	int tick_counter = 0;
	bool shortcut_state = true;
	int gorche;
	int write_counter = 0;
	FILE * fp;
	fp = fopen ("distribution.txt","w");

	balls_init(balls, n, R, width, height, a, b, v_max);
	//balls_init_from_file(balls, n, R, width, height, a, b, v_max);
	
	while (running) {
		tick_counter++;
        draw_table(balls, width, height, a, b, top_border, bottom_border, right_border, left_border, R, n, l);
		if(running){
			//scanf ("%d",&gorche);
			enum State state = GAME_ON;
			tick += shortcut_step(balls, top_border, bottom_border, left_border, right_border, R, mu, tick, delta_t, n, k, l, &state, width, height, a, b) * delta_t;
			if(isnan(tick)) state = check_table(balls, n, top_border, bottom_border, left_border, right_border, v_max, tick, mu, l);
			if(state == BALL_BEYOND_TABLE && running){
				printf("BALL_BEYOND_TABLE\n");
				running = false;
			}
			if(state == LACK_OF_ENERGY && running){
				printf("LACK_OF_ENERGY\n");
				running = false;
			}
			mechanics_step(balls, top_border, bottom_border, left_border, right_border, R, mu, tick, delta_t, n, k, l, &n_count, &d_count);
			if(all_balls_in_move(balls, n)) write_velocity_distribution(fp, balls, n, &write_counter);
		}
		
		tick += delta_t;
		draw_table(balls, width, height, a, b, top_border, bottom_border, right_border, left_border, R, n, l);
		//printf("\n");
	}

	al_destroy_display(display);
	fclose (fp);

	return 0;
}
