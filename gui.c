#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>
#include <allegro5/allegro_image.h>
#include "hot_zone_kill.h"


void draw_table(struct Ball balls[], struct Params *params){
	al_flip_display();
	al_draw_filled_rectangle(0, 0, params->width, params->height, al_map_rgb(0, 0, 0));
	int thickness = 5;
	int top_border = params->top_border;
	int bottom_border = params->bottom_border;
	int right_border = params->right_border;
	int left_border = params->left_border;
	double l = params->l;
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
	if(params->width > params->height){
		al_draw_line((params->width-params->l)/2, top_border, (params->width+params->l)/2, top_border, al_map_rgb(0, 0, 0), thickness);
		al_draw_line((params->width-params->l)/2, bottom_border, (params->width+params->l)/2, bottom_border, al_map_rgb(0, 0, 0), thickness);
	}
	for (int i = 0; i < params->n; ++i)
	{
		if(!balls[i].color.filled) al_draw_ellipse(round(balls[i].x), round(balls[i].y), params->R, params->R, al_map_rgb(balls[i].color.r, balls[i].color.g, balls[i].color.b), 1);
		else al_draw_filled_ellipse(round(balls[i].x), round(balls[i].y), params->R, params->R, al_map_rgb(balls[i].color.r, balls[i].color.g, balls[i].color.b));
	}
}

void visualize(struct Ball balls[], struct Params *params, double t0, double tick, double delta_t){
	int mi = 1;
	while(t0+mi*delta_t < tick){
		double dx[params->n];
		double dy[params->n];
		for(int i=0; i<params->n; i++){
			double v_x, v_y;
			double mu_x, mu_y;
			if(params->motion_mode == UNIFORMLY_DECELERATED){
				double sin_a = sinus(balls[i].v_x, balls[i].v_y);
				double cos_a = cosinus(balls[i].v_x, balls[i].v_y);
				mu_x = params->mu*cos_a;
				mu_y = params->mu*sin_a;
				v_x = balls[i].v_x - params->mu*cos_a*(t0 - balls[i].tick_base);
				v_y = balls[i].v_y - params->mu*sin_a*(t0 - balls[i].tick_base);
				//printf("%lf - %lf\n", balls[i].v_x - params->mu*cos_a*(tick - balls[i].tick_base));
			}
			else{
				mu_x = params->mu;
				mu_y = params->mu;
				v_x = balls[i].v_x;
				v_y = balls[i].v_y;
			}
			dx[i] = movement_integral(v_x, mu_x, t0+mi*delta_t-balls[i].tick_base, t0-balls[i].tick_base, params->motion_mode);
			dy[i] = movement_integral(v_y, mu_y, t0+mi*delta_t-balls[i].tick_base, t0-balls[i].tick_base, params->motion_mode);
			//printf("%lf\n", dx[i]);
			balls[i].x += dx[i];
			balls[i].y += dy[i];
		}
		draw_table(balls, params);
		for(int i=0; i<params->n; i++){
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
	
	struct Params *params = malloc(sizeof(struct Params));
	params->width = 2*640;
	params->height = 2*480;
	params->a = 2*600;
	params->b = 2*400;
	params->R = 8.0;
	params->top_border = (params->height - params->b) / 2;
	params->bottom_border = (params->height + params->b) / 2;
	params->right_border = (params->width + params->a) / 2;
	params->left_border = (params->width - params->a) / 2;
	params->mu = 0.00018;
	params->n = 7;
	params->l = 60.0;
	params->k = 1;
	params->delta_t = 0.001;
	params->v_max = 1*sqrt(2);
	//params->motion_mode = PROPORTIONAL_TO_VELOCITY;
	params->motion_mode = UNIFORMLY_DECELERATED;
	display = al_create_display(params->width, params->height);
	
	struct Ball balls[params->n];
	double tick = 0.0;
	int n_count = 0;
	int d_count = 0;
	int frame_counter = 0;
	int tick_counter = 0;
	bool running = true;
	int gorche;
	int write_counter = 0;
	FILE * fp;
	fp = fopen ("distribution.txt","w");

	balls_init(balls, params);
	//balls_init_from_file(balls, params);
	while (running) {
		tick_counter++;
        draw_table(balls, params);

		enum State state = GAME_ON;
		int putting_out_id = -1;
		tick += shortcut_step(balls, params, tick, &state, &putting_out_id) * params->delta_t;
		if(params->motion_mode == UNIFORMLY_DECELERATED && putting_out_id != -1){
			tick += ud_putting_out(balls, params, putting_out_id, tick);
			continue;
		}
		if(isnan(tick)) state = check_table(balls, params, tick);
		if(state == BALL_BEYOND_TABLE && running){
			printf("BALL_BEYOND_TABLE\n");
			running = false;
		}
		if(state == LACK_OF_ENERGY && running){
			printf("LACK_OF_ENERGY\n");
			running = false;
		}
		if(running){
			mechanics_step(balls, params, tick, &n_count, &d_count);
			//if(all_balls_in_move(balls, params->n)) write_velocity_distribution(fp, balls, params->n, &write_counter);
			tick += params->delta_t;
		}
		draw_table(balls, params);
	}

	al_destroy_display(display);
	fclose (fp);

	return 0;
}
