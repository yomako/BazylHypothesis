#ifndef HEADER_FILE
#define HEADER_FILE


#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

struct Color
{
	int r;
	int g;
	int b;
	bool filled;
};

struct Ball
{	
	struct Color color;
	double x_0;
	double x_00;
	double y_0;
	double y_00;
	double x;
	double y;
	double v_x;
	double v_y;
	double v_x00;
	double v_y00;
	double tick_base;
	double x_corner;
	double y_corner;
	bool white;
	bool black;
	bool pocket_available;
	bool corner_available;
	bool border_available;
};

struct Params
{
	int width;
	int height;
	int a;
	int b;
	int top_border;
	int bottom_border;
	int right_border;
	int left_border;
	int n;
	double mu;
	double l;
	double k;
	double delta_t;
	double v_max;
	double R;
};

enum State {BALL_BEYOND_TABLE, LACK_OF_ENERGY, GAME_ON, ERROR};
enum Return_mode {POCKET, CORNER, BORDER, COLLISION};

void shuffle(int *a, int *b){
	int temp = *a;
	*a = *b;
	*b = temp;
}

void color_init(struct Color *color, int r, int g, int b, bool filled){
	color->r = r;
	color->g = g;
	color->b = b;
	color->filled = filled;
}

void balls_init(struct Ball *balls, struct Params *params){
	struct Color balls_colors[16];
	color_init(&balls_colors[0], 255, 255, 255, true);
	color_init(&balls_colors[1], 255, 255, 255, false);
	color_init(&balls_colors[2], 255, 255, 0, false);
	color_init(&balls_colors[3], 255, 255, 0, true);
	color_init(&balls_colors[4], 0, 0, 255, true);
	color_init(&balls_colors[5], 0, 0, 255, false);
	color_init(&balls_colors[6], 255, 0, 0, true);
	color_init(&balls_colors[7], 255, 0, 0, false);
	color_init(&balls_colors[8], 128, 0, 128, true);
	color_init(&balls_colors[9], 128, 0, 128, false);
	color_init(&balls_colors[10], 255, 165, 0, true);
	color_init(&balls_colors[11], 255, 165, 0, false);
	color_init(&balls_colors[12], 0, 255, 0, true);
	color_init(&balls_colors[13], 0, 255, 0, false);
	color_init(&balls_colors[14], 165, 42, 42, true);
	color_init(&balls_colors[15], 165, 42, 42, false);

	double theta = 2 * M_PI * (double)rand() / RAND_MAX;
	//printf("%lf\n", theta);
	for (int i = 0; i < params->n; ++i)
	{
		bool conflict;
		do{
			//printf("width=%d, height=%d, a=%d, b=%d\n", width, height, a, b);
			balls[i].x = (params->width - params->a) / 2 + params->R + (double)rand() / RAND_MAX * (params->width - 2 * ((params->width - params->a) / 2 + params->R));
			balls[i].y = (params->height - params->b) / 2 + params->R + (double)rand() / RAND_MAX * (params->height - 2 * ((params->height - params->b) / 2 + params->R));
			conflict = false;
			for (int j = 0; j < i; ++j)
			{
				if((balls[i].x - balls[j].x)*(balls[i].x - balls[j].x) + (balls[i].y - balls[j].y)*(balls[i].y - balls[j].y) <= 4*params->R*params->R) conflict = true;
				//printf("i=(%lf, %lf), j=(%lf, %lf)\n", balls[i].x, balls[i].y, balls[j].x, balls[j].y);
				//printf("%lf | %lf\n", (balls[i].x - balls[j].x)*(balls[i].x - balls[j].x) + (balls[i].y - balls[j].y)*(balls[i].y - balls[j].y), 4*R*R);
			}
			//printf("%d\n", conflict);
			//printf("\n");
		}while(conflict);
		//printf("%lf\n", balls[i].x);
		//printf("%lf\n", balls[i].y);
		balls[i].color = balls_colors[0];
		if(i==0){
			balls[i].white = true;
			balls[i].black = false;
		}
		else if(i==1){
			balls[i].white = false;
			balls[i].black = true;
		}
		else{
			balls[i].white = false;
			balls[i].black = false;
		}
		if(balls[i].white){
			balls[i].v_x = params->v_max * cos(theta);
			balls[i].v_y = params->v_max * sin(theta);
		}
		else{
			balls[i].v_x = 0.0;
			balls[i].v_y = 0.0;
		}
		balls[i].tick_base = 0.0;
		balls[i].x_00 = balls[i].x_0;
		balls[i].y_00 = balls[i].y_0;
		balls[i].v_x00 = balls[i].v_x;
		balls[i].v_y00 = balls[i].v_y;
	}
}


void balls_init_from_file(struct Ball *balls, struct Params *params){
	struct Color balls_colors[16];
	color_init(&balls_colors[0], 255, 255, 255, true);
	color_init(&balls_colors[1], 255, 255, 255, false);
	color_init(&balls_colors[2], 255, 255, 0, false);
	color_init(&balls_colors[3], 255, 255, 0, true);
	color_init(&balls_colors[4], 0, 0, 255, true);
	color_init(&balls_colors[5], 0, 0, 255, false);
	color_init(&balls_colors[6], 255, 0, 0, true);
	color_init(&balls_colors[7], 255, 0, 0, false);
	color_init(&balls_colors[8], 128, 0, 128, true);
	color_init(&balls_colors[9], 128, 0, 128, false);
	color_init(&balls_colors[10], 255, 165, 0, true);
	color_init(&balls_colors[11], 255, 165, 0, false);
	color_init(&balls_colors[12], 0, 255, 0, true);
	color_init(&balls_colors[13], 0, 255, 0, false);
	color_init(&balls_colors[14], 165, 42, 42, true);
	color_init(&balls_colors[15], 165, 42, 42, false);

	int color_indices[14];
	for(int i = 0; i < 14; ++i){
		color_indices[i] = i + 2;
	}

	for (int i = 0; i < 100; ++i)
	{
		shuffle(&color_indices[rand()%14], &color_indices[rand()%14]);
	}
	int indices[16];
	indices[0] = 0;
	indices[1] = 1;
	for(int i = 0; i < 14; ++i){
		indices[i+2] = color_indices[i];
	}

	FILE * fp;
	size_t len = 0;
    char * line = NULL;

	fp = fopen("init.txt", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    getline(&line, &len, fp);
    double theta;
    sscanf(line, "%lf", &theta);

	for (int i = 0; i < params->n; ++i)
	{
		getline(&line, &len, fp);
		sscanf(line, "%lf", &balls[i].x);
		getline(&line, &len, fp);
		sscanf(line, "%lf", &balls[i].y);
		balls[i].color = balls_colors[indices[i]];
		if(i==0){
			balls[i].white = true;
			balls[i].black = false;
		}
		else if(i==1){
			balls[i].white = false;
			balls[i].black = true;
		}
		else{
			balls[i].white = false;
			balls[i].black = false;
		}
		if(balls[i].white){
			balls[i].v_x = params->v_max * cos(theta);
			balls[i].v_y = params->v_max * sin(theta);
		}
		else{
			balls[i].v_x = 0.0;
			balls[i].v_y = 0.0;
		}
		balls[i].tick_base = 0.0;
		balls[i].x_00 = balls[i].x_0;
		balls[i].y_00 = balls[i].y_0;
		balls[i].v_x00 = balls[i].v_x;
		balls[i].v_y00 = balls[i].v_y;
	}

	fclose(fp);
    if (line)
        free(line);
}

double movement_integral(double v_0, double mu, double upper_limes, double lower_limes){
	if(mu == 0.0) return v_0 * (upper_limes - lower_limes);
	else return v_0/mu*(exp(-mu*lower_limes) - exp(-mu*upper_limes));
}

void horizontal_return(struct Ball *ball_j, struct Params *params, double border, double tick){
	double c;
	if(params->mu == 0.0) c = (border-ball_j->y_0) / ball_j->v_y; 
	else c = 1/params->mu * log(1/(1-params->mu*(border-ball_j->y_0)/(ball_j->v_y)*exp(params->mu*(tick-ball_j->tick_base))));
	if(c < 0) return;
	double x_z = ball_j->x_0 + movement_integral(ball_j->v_x, params->mu, tick-ball_j->tick_base+c, tick-ball_j->tick_base);
	if(x_z > (params->left_border + params->l/sqrt(2)) && x_z < (params->right_border - params->l/sqrt(2))){
		ball_j->y = ball_j->y_0 + movement_integral(ball_j->v_y, params->mu, tick-ball_j->tick_base+c, tick-ball_j->tick_base);
		ball_j->v_y *= -1;
		ball_j->y = ball_j->y + movement_integral(ball_j->v_y, params->mu, tick-ball_j->tick_base+params->delta_t, tick-ball_j->tick_base+c);
		ball_j->x = ball_j->x_0 + movement_integral(ball_j->v_x, params->mu, tick-ball_j->tick_base+params->delta_t, tick-ball_j->tick_base);
	}
	else{
		ball_j->y = ball_j->y_0 + movement_integral(ball_j->v_y, params->mu, tick-ball_j->tick_base+params->delta_t, tick-ball_j->tick_base);
		ball_j->x = ball_j->x_0 + movement_integral(ball_j->v_x, params->mu, tick-ball_j->tick_base+params->delta_t, tick-ball_j->tick_base);
	}
}

void vertical_return(struct Ball *ball_j, struct Params *params, double border, double tick){
	double c;
	if(params->mu == 0.0) c = (border-ball_j->x_0) / ball_j->v_x; 
	else c = 1/params->mu * log(1/(1-params->mu*(border-ball_j->x_0)/ball_j->v_x*exp(params->mu*(tick-ball_j->tick_base))));
	if(c < 0) return;
	double y_z = ball_j->y_0 + movement_integral(ball_j->v_y, params->mu, tick-ball_j->tick_base+c, tick-ball_j->tick_base);
	if(y_z > (params->top_border + params->l/sqrt(2)) && y_z < (params->bottom_border - params->l/sqrt(2))){
		ball_j->x = ball_j->x_0 + movement_integral(ball_j->v_x, params->mu, tick-ball_j->tick_base+c, tick-ball_j->tick_base);
		ball_j->v_x *= -1;
		ball_j->y = ball_j->y_0 + movement_integral(ball_j->v_y, params->mu, tick-ball_j->tick_base+params->delta_t, tick-ball_j->tick_base);
		ball_j->x = ball_j->x + movement_integral(ball_j->v_x, params->mu, tick-ball_j->tick_base+params->delta_t, tick-ball_j->tick_base+c);
	}
	else{
		ball_j->y = ball_j->y_0 + movement_integral(ball_j->v_y, params->mu, tick-ball_j->tick_base+params->delta_t, tick-ball_j->tick_base);
		ball_j->x = ball_j->x_0 + movement_integral(ball_j->v_x, params->mu, tick-ball_j->tick_base+params->delta_t, tick-ball_j->tick_base);
	}
}

void corner_return(struct Ball *ball_j, struct Params *params, double tick){
	double c;
	if(params->mu == 0.0){
		double x_p = ball_j->x_0 - ball_j->x_corner;
		double y_p = ball_j->y_0 - ball_j->y_corner;
		double A = ball_j->v_x*ball_j->v_x + ball_j->v_y*ball_j->v_y;
		double B = 2*(ball_j->v_x*x_p + ball_j->v_y*y_p);
		double C = x_p*x_p + y_p*y_p - params->R*params->R;
		double delta = B*B - 4*A*C;
		if(delta < 0) return;
		c = (-B - sqrt(delta))/2/A;
	}
	else{
		double v_x = ball_j->v_x*exp(-params->mu*(tick - ball_j->tick_base));
		double v_y = ball_j->v_y*exp(-params->mu*(tick - ball_j->tick_base));
		double x_p = ball_j->x_0 + v_x/params->mu - ball_j->x_corner;
		double y_p = ball_j->y_0 + v_y/params->mu - ball_j->y_corner;
		double A = (v_x*v_x + v_y*v_y)/params->mu/params->mu;
		double B = -2*(x_p*v_x + y_p*v_y)/params->mu;
		double C = x_p*x_p + y_p*y_p - params->R*params->R;
		double delta = B*B - 4*A*C;
		if(delta < 0) return;
		double g = (-B + sqrt(delta)) / 2 / A;
		c = -1/params->mu*log(g);
	}
	ball_j->x = ball_j->x_0 + movement_integral(ball_j->v_x, params->mu, tick-ball_j->tick_base+c, tick-ball_j->tick_base);
	ball_j->y = ball_j->y_0 + movement_integral(ball_j->v_y, params->mu, tick-ball_j->tick_base+c, tick-ball_j->tick_base);
	double v_xp = ball_j->v_x*exp(-params->mu*(tick - ball_j->tick_base + c));
	double v_yp = ball_j->v_y*exp(-params->mu*(tick - ball_j->tick_base + c));
	double v = sqrt(v_xp*v_xp+v_yp*v_yp);
	double alpha = acos((ball_j->x_corner - ball_j->x)/params->R);
	double beta = acos(v_xp/v);
	double vq = v * cos(beta - alpha);
	double vp = v * sin(beta - alpha);
	ball_j->v_x = vq * cos(alpha + M_PI) + vp * cos(alpha + M_PI/2);
	ball_j->v_y = vq * sin(alpha + M_PI) + vp * sin(alpha + M_PI/2);
	ball_j->tick_base = tick + c;
	ball_j->x += movement_integral(ball_j->v_x, params->mu, params->delta_t-c, 0);
	ball_j->y += movement_integral(ball_j->v_y, params->mu, params->delta_t-c, 0);
}

void collision(struct Ball *ball_j, struct Ball *ball_k, struct Params *params, double tick){
	double R = params->R;
	double mu = params->mu;
	double k = params->k;
	double delta_t = params->delta_t;
	// 1)
	double x_0dr = ball_j->x_0 - ball_k->x_0;
	double y_0dr = ball_j->y_0 - ball_k->y_0;
	// 2)
	double v_oxdr = ball_j->v_x*exp(-mu*(tick-ball_j->tick_base)) - ball_k->v_x*exp(-mu*(tick-ball_k->tick_base));
	double v_oydr = ball_j->v_y*exp(-mu*(tick-ball_j->tick_base)) - ball_k->v_y*exp(-mu*(tick-ball_k->tick_base));
	// 3)
	double gamma = v_oxdr*v_oxdr + v_oydr*v_oydr;
	double beta = 2*(v_oxdr*x_0dr+v_oydr*y_0dr);
	double alpha = x_0dr*x_0dr + y_0dr*y_0dr - 4*R*R;
	// 4)
	double sqrt_delta = sqrt(beta*beta - 4*gamma*alpha);
	// 5)
	double c;
	if(mu == 0.0){
		c = -(beta + sqrt_delta) / 2 / gamma;
	}
	else{
		double w = -(beta + sqrt_delta) / 2 / gamma;
		c = log(1/(1-mu*w)) / mu;
	}
	// 7)

	double delta_x_cdj = movement_integral(ball_j->v_x, mu, tick-ball_j->tick_base+c, tick-ball_j->tick_base);
	double delta_y_cdj = movement_integral(ball_j->v_y, mu, tick-ball_j->tick_base+c, tick-ball_j->tick_base);
	double delta_x_cdk = movement_integral(ball_k->v_x, mu, tick-ball_k->tick_base+c, tick-ball_k->tick_base);
	double delta_y_cdk = movement_integral(ball_k->v_y, mu, tick-ball_k->tick_base+c, tick-ball_k->tick_base);
	// 8)
	double xdj = ball_j->x_0 + delta_x_cdj;
	double ydj = ball_j->y_0 + delta_y_cdj;
	double xdk = ball_k->x_0 + delta_x_cdk;
	double ydk = ball_k->y_0 + delta_y_cdk;
	// 9)
	double sin_phi = (xdk - xdj) / sqrt((xdk - xdj)*(xdk - xdj) + (ydk - ydj)*(ydk - ydj));
	double cos_phi = (ydj - ydk) / sqrt((xdk - xdj)*(xdk - xdj) + (ydk - ydj)*(ydk - ydj));

	double v_psidj = (ball_j->v_x*cos_phi + ball_j->v_y*sin_phi)*exp(-mu*(tick-ball_j->tick_base+c));
	double v_etadj = (ball_j->v_y*cos_phi - ball_j->v_x*sin_phi)*exp(-mu*(tick-ball_j->tick_base+c));
	double v_psidk = (ball_k->v_x*cos_phi + ball_k->v_y*sin_phi)*exp(-mu*(tick-ball_k->tick_base+c));
	double v_etadk = (ball_k->v_y*cos_phi - ball_k->v_x*sin_phi)*exp(-mu*(tick-ball_k->tick_base+c));
	////////////////////////////////////////
	double w_etadj = (1-k)/2*v_etadj+(k+1)/2*v_etadk;
	double w_etadk = (k+1)/2*v_etadj+(1-k)/2*v_etadk;

	double sin_thetadj;
	double cos_thetadj;
	double sin_thetadk;
	double cos_thetadk;
	if(ball_j->v_y == 0.0 && ball_j->v_x == 0.0){
		sin_thetadj = 1 / sqrt(2);
		cos_thetadj = 1 / sqrt(2);
	}
	else{
		sin_thetadj = ball_j->v_y / sqrt(ball_j->v_x*ball_j->v_x + ball_j->v_y*ball_j->v_y);
		cos_thetadj = ball_j->v_x / sqrt(ball_j->v_x*ball_j->v_x + ball_j->v_y*ball_j->v_y);
	}
	if(ball_k->v_y == 0.0 && ball_k->v_x == 0.0){
		sin_thetadk = 1 / sqrt(2);
		cos_thetadk = 1 / sqrt(2);
	}
	else{
		sin_thetadk = ball_k->v_y / sqrt(ball_k->v_x*ball_k->v_x + ball_k->v_y*ball_k->v_y);
		cos_thetadk = ball_k->v_x / sqrt(ball_k->v_x*ball_k->v_x + ball_k->v_y*ball_k->v_y);
	}

	double w_psidj = sqrt(ball_j->v_x*ball_j->v_x + ball_j->v_y*ball_j->v_y)*exp(-mu*(tick-ball_j->tick_base+c))*(sin_thetadj*sin_phi+cos_phi*cos_thetadj);
	double w_psidk = sqrt(ball_k->v_x*ball_k->v_x + ball_k->v_y*ball_k->v_y)*exp(-mu*(tick-ball_k->tick_base+c))*(sin_thetadk*sin_phi+cos_phi*cos_thetadk);

	//////////////////////////
	ball_j->v_x = w_psidj*cos_phi - w_etadj*sin_phi;
	ball_j->v_y = w_psidj*sin_phi + w_etadj*cos_phi;
	ball_k->v_x = w_psidk*cos_phi - w_etadk*sin_phi;
	ball_k->v_y = w_psidk*sin_phi + w_etadk*cos_phi;

	// 10)
	ball_j->tick_base = tick + c;
	ball_k->tick_base = tick + c;
	// 11)
	ball_j->x = xdj + movement_integral(ball_j->v_x, mu, delta_t-c, 0);
	ball_j->y = ydj + movement_integral(ball_j->v_y, mu, delta_t-c, 0);
	ball_k->x = xdk + movement_integral(ball_k->v_x, mu, delta_t-c, 0);
	ball_k->y = ydk + movement_integral(ball_k->v_y, mu, delta_t-c, 0);
	if((ball_j->x-ball_k->x)*(ball_j->x-ball_k->x)+(ball_j->y-ball_k->y)*(ball_j->y-ball_k->y) <= 4*R*R){
		printf("nie spełnia=%lf\n", (ball_j->x-ball_k->x)*(ball_j->x-ball_k->x)+(ball_j->y-ball_k->y)*(ball_j->y-ball_k->y));
	}

}

void putting_out(struct Ball balls[], int n, double mu, double tick){
	for (int i = 0; i < n; ++i)
	{
		double dx = balls[i].v_x/mu*(exp(-mu*(tick-balls[i].tick_base)));
		double dy = balls[i].v_y/mu*(exp(-mu*(tick-balls[i].tick_base)));
		balls[i].x += dx;
		balls[i].y += dy;
	}
}

double border_shortcut(struct Ball *ball_j, struct Params *params, double q0, double border, double tick, char mode){
	double v0q;
	if(mode == 'v') v0q = ball_j->v_y*exp(-params->mu*(tick - ball_j->tick_base));
	else v0q = ball_j->v_x*exp(-params->mu*(tick - ball_j->tick_base));
	double t;
	if(params->mu == 0.0) t = (border - q0) / v0q;
	else t = -1/params->mu*log(1 - params->mu/v0q*(border - q0));
	if(mode == 'v'){
		double x1 = ball_j->x + movement_integral(ball_j->v_x, params->mu, tick-ball_j->tick_base+t, tick-ball_j->tick_base);
		if(x1 < params->left_border + params->l/sqrt(2) || (x1 > (params->right_border + params->left_border)/2-params->l/2 && 
			x1 < (params->right_border + params->left_border)/2+params->l/2) || x1 > params->right_border - params->l/sqrt(2)) return nan("1");
	}
	else{
		double y1 = ball_j->y + movement_integral(ball_j->v_y, params->mu, tick-ball_j->tick_base+t, tick-ball_j->tick_base);
		if(y1 < params->top_border + params->l/sqrt(2) || y1 > params->bottom_border - params->l/sqrt(2)) return nan("1");
	}
	if(t >= 0){
		ball_j->border_available = true;
		return t;
	}
	else return nan("1");
}


double find_border_crossing(double p, double q, double R, double border, double p_co){
	double p_b1 = p + sqrt(R*R - (border - q)*(border - q));
	double p_b2 = p - sqrt(R*R - (border - q)*(border - q));
	double p_b;
	if(abs(p_b1 - p_co) > abs(p_b2 - p_co)) p_b = p_b1;
	else p_b = p_b2;
	return p_b;
}


bool check_border_crossing(struct Ball *ball_j, struct Params *params, double tick, double x_co, double y_co, double t){
	double x_p = ball_j->x + movement_integral(ball_j->v_x, params->mu, tick-ball_j->tick_base+t, tick-ball_j->tick_base);
	double y_p = ball_j->y + movement_integral(ball_j->v_y, params->mu, tick-ball_j->tick_base+t, tick-ball_j->tick_base);
	bool border_crossing = false;
	int top_border = params->top_border;
	int bottom_border = params->bottom_border;
	int right_border = params->right_border;
	int left_border = params->left_border;
	double l = params->l;
	double R = params->R;
	if(((x_co == left_border) && (y_co == (top_border+l/sqrt(2)))) || ((x_co == left_border) && (y_co == (bottom_border-l/sqrt(2))))){
		double y_b = find_border_crossing(y_p, x_p, R, left_border, y_co);
		if((y_b > (top_border+l/sqrt(2))) && (y_b < (bottom_border-l/sqrt(2)))) border_crossing = true;
	}
	else if(((x_co == (left_border+l/sqrt(2))) && (y_co == top_border)) || ((x_co == ((right_border + left_border)/2-l/2)) && (y_co == top_border))){
		double x_b = find_border_crossing(x_p, y_p, R, top_border, x_co);
		if((x_b > (left_border+l/sqrt(2))) && (x_b < ((right_border + left_border)/2-l/2))) border_crossing = true;
	}
	else if(((x_co == (left_border+l/sqrt(2))) && (y_co == bottom_border)) || ((x_co == ((right_border + left_border)/2-l/2)) && (y_co == bottom_border))){
		double x_b = find_border_crossing(x_p, y_p, R, bottom_border, x_co);
		if((x_b > (left_border+l/sqrt(2))) && (x_b < ((right_border + left_border)/2-l/2))) border_crossing = true;
	}
	else if(((x_co == ((right_border + left_border)/2+l/2)) && (y_co == bottom_border)) || ((x_co == (right_border-l/sqrt(2))) && (y_co == bottom_border))){
		double x_b = find_border_crossing(x_p, y_p, R, bottom_border, x_co);
		if((x_b > ((right_border + left_border)/2+l/2)) && (x_b < (right_border-l/sqrt(2)))) border_crossing = true;
	}
	else if(((x_co == right_border) && (y_co == (bottom_border-l/sqrt(2)))) || ((x_co == right_border) && (y_co == (top_border+l/sqrt(2))))){
		double y_b = find_border_crossing(y_p, x_p, R, right_border, y_co);
		if((y_b > (top_border+l/sqrt(2))) && (y_b < (bottom_border-l/sqrt(2)))) border_crossing = true;
	}
	else if(((x_co == (right_border-l/sqrt(2))) && (y_co == top_border)) || ((x_co == ((right_border + left_border)/2+l/2)) && (y_co == top_border))){
		double x_b = find_border_crossing(x_p, y_p, R, top_border, x_co);
		if((x_b > ((right_border + left_border)/2+l/2)) && (x_b < (right_border-l/sqrt(2)))) border_crossing = true;
	}
	return border_crossing;
}


double corner_shortcut(struct Ball *ball_j, struct Params *params, double tick, double x_co, double y_co){
	if(params->mu == 0.0){
		double x_p = ball_j->x - x_co;
		double y_p = ball_j->y - y_co;
		double A = ball_j->v_x*ball_j->v_x + ball_j->v_y*ball_j->v_y;
		double B = 2*(ball_j->v_x*x_p + ball_j->v_y*y_p);
		double C = x_p*x_p + y_p*y_p - params->R*params->R;
		double delta = B*B - 4*A*C;
		if(delta < 0) return nan("1");
		double t = (-B - sqrt(delta))/2/A;
		bool border_crossing = check_border_crossing(ball_j, params, tick, x_co, y_co, t);
		if(!border_crossing && !isnan(t) && t > 0){
			ball_j->corner_available = true;
			ball_j->x_corner = x_co;
			ball_j->y_corner = y_co;
			return t;
		}
		else return nan("1");
	}
	else{
		double v_x = ball_j->v_x*exp(-params->mu*(tick - ball_j->tick_base));
		double v_y = ball_j->v_y*exp(-params->mu*(tick - ball_j->tick_base));
		double x_p = ball_j->x + v_x/params->mu - x_co;
		double y_p = ball_j->y + v_y/params->mu - y_co;
		double A = (v_x*v_x + v_y*v_y)/params->mu/params->mu;
		double B = -2*(x_p*v_x + y_p*v_y)/params->mu;
		double C = x_p*x_p + y_p*y_p - params->R*params->R;
		double delta = B*B - 4*A*C;
		if(delta < 0) return nan("1");
		double g = (-B + sqrt(delta)) / 2 / A;
		double t = -1/params->mu*log(g);
		bool border_crossing = check_border_crossing(ball_j, params, tick, x_co, y_co, t);
		if(!border_crossing && !isnan(t) && t > 0){
			ball_j->corner_available = true;
			ball_j->x_corner = x_co;
			ball_j->y_corner = y_co;
			return t;
		}
		else return nan("1");
	}
}


double balls_shortcut(struct Params *params, double x_10, double y_10, double v_1x, double v_1y, double x_20, double y_20, double v_2x, double v_2y){
	double dx = x_10 - x_20;
	double dy = y_10 - y_20;
	if(params->mu == 0.0){
		double dvx = v_1x - v_2x;
		double dvy = v_1y - v_2y;
		double A = dvx*dvx + dvy*dvy;
		double B = 2*(dx*dvx + dy*dvy);
		double C = dx*dx + dy*dy - 4*params->R*params->R;
		double delta = B*B - 4*A*C;
		if(delta < 0) return nan("1");
		double t = (-B - sqrt(delta))/2/A;
		if(t >= 0){
			return t;
		}
		else return nan("1");
	}
	else{
		double dvx = 1/params->mu*(v_1x - v_2x);
		double dvy = 1/params->mu*(v_1y - v_2y);
		double alpha = dx + dvx;
		double beta = dy + dvy;
		double C = alpha*alpha + beta*beta - 4*params->R*params->R;
		double B = -2*(alpha*dvx + beta*dvy);
		double A = dvx*dvx + dvy*dvy;
		double delta = B*B - 4*A*C;
		if(delta < 0) return nan("1");
		double t = -log(-(B-sqrt(delta))/2/A)/params->mu;
		if(t >= 0){
			return t;
		}
		else return nan("1");
	}
}


double pocket_shortcut(double g, double b, double xc, struct Ball *ball_j, struct Params *params, double tick){
	double v_x = ball_j->v_x*exp(-params->mu*(tick - ball_j->tick_base));
	double v_y = ball_j->v_y*exp(-params->mu*(tick - ball_j->tick_base));
	double x3 = (ball_j->y-v_y/v_x*ball_j->x - b) / (g - v_y/v_x);
	double x1 = (ball_j->y-params->R*sqrt(1+(v_y/v_x)*(v_y/v_x))-v_y/v_x*ball_j->x - b) / (g - v_y/v_x);
	double x2 = (ball_j->y+params->R*sqrt(1+(v_y/v_x)*(v_y/v_x))-v_y/v_x*ball_j->x - b) / (g - v_y/v_x);
	
	if(((1+g*g)*(x1 - xc)*(x1 - xc) <= params->l*params->l/4) && ((1+g*g)*(x2 - xc)*(x2 - xc) <= params->l*params->l/4)){
		double t;
		if(params->mu == 0.0) t = (x3 - ball_j->x) / v_x;
		else t = -log(1 - params->mu*(x3 - ball_j->x)/(v_x))/params->mu;
		if(isnan(t)) return nan("1");
		if(t >= 0){
			ball_j->pocket_available = true;
			return t;
		}
		else return nan("1");
	}
	else return nan("1");
}

double find_lowest_m(double a, double b, bool *pocket_is_lowest, enum Return_mode return_mode, struct Ball *ball_j){
	if(isnan(a) && !isnan(b)) return b;
	if(!isnan(a) && isnan(b)){
		if(return_mode == POCKET) *pocket_is_lowest = true;
		else if(return_mode == CORNER || return_mode == COLLISION) *pocket_is_lowest = false;
		else if(return_mode == BORDER){
			*pocket_is_lowest = false;
			ball_j->corner_available = false;
		}
		return a;
	}
	if(isnan(a) && isnan(b)) return b;
	if(!isnan(a) && !isnan(b)){
		if(a < b){
			if(return_mode == POCKET) *pocket_is_lowest = true;
			else if(return_mode == CORNER || return_mode == COLLISION) *pocket_is_lowest = false;
			else if(return_mode == BORDER){
				*pocket_is_lowest = false;
				ball_j->corner_available = false;
			}
			return a;
		}
		if(a > b) return b;
	}
}

double find_lowest(double a, double b, bool *pocket_is_lowest, bool pocket_is_lowest_a){
	if(isnan(a) && !isnan(b)) return b;
	if(!isnan(a) && isnan(b)){
		*pocket_is_lowest = pocket_is_lowest_a;
		return a;
	}
	if(isnan(a) && isnan(b)) return b;
	if(!isnan(a) && !isnan(b)){
		if(a < b){
			*pocket_is_lowest = pocket_is_lowest_a;
			return a;
		}
		if(a > b) return b;
	}
}

bool is_lower(double a, double lowest){
	if(isnan(a) && isnan(lowest)) return false;
	if(!isnan(a) && isnan(lowest)) return true;
	if(isnan(a) && !isnan(lowest)) return false;
	if(!isnan(a) && !isnan(lowest)){
		if(a < lowest) return true;
		else return false;
	}
}


double lowest_floor_operation(double a, double b){
	if(isnan(a)) return a;
	else return floor(a / b);
}


double autonomical_lowest(struct Ball *ball_j, struct Params *params, double tick, bool *pocket_is_lowest){
	double lowest = nan("1");
	ball_j->pocket_available = false;
	ball_j->corner_available = false;
	ball_j->border_available = false;
	*pocket_is_lowest = false;
	int top_border = params->top_border;
	int bottom_border = params->bottom_border;
	int right_border = params->right_border;
	int left_border = params->left_border;
	double l = params->l;
	double R = params->R;
	double delta_t = params->delta_t;
	lowest = find_lowest_m(floor(pocket_shortcut(-1, top_border+(left_border + l/sqrt(2)), left_border + l/2/sqrt(2), ball_j, params, tick) / delta_t), lowest, pocket_is_lowest, POCKET, ball_j);
	lowest = find_lowest_m(floor(pocket_shortcut(1, bottom_border-(left_border + l/sqrt(2)), left_border + l/2/sqrt(2), ball_j, params, tick) / delta_t), lowest, pocket_is_lowest, POCKET, ball_j);
	lowest = find_lowest_m(floor(pocket_shortcut(0, bottom_border, (right_border + left_border)/2, ball_j, params, tick) / delta_t), lowest, pocket_is_lowest, POCKET, ball_j);
	lowest = find_lowest_m(floor(pocket_shortcut(-1, bottom_border+(right_border - l/sqrt(2)), right_border - l/2/sqrt(2), ball_j, params, tick) / delta_t), lowest, pocket_is_lowest, POCKET, ball_j);
	lowest = find_lowest_m(floor(pocket_shortcut(1, top_border-(right_border - l/sqrt(2)), right_border - l/2/sqrt(2), ball_j, params, tick) / delta_t), lowest, pocket_is_lowest, POCKET, ball_j);
	lowest = find_lowest_m(floor(pocket_shortcut(0, top_border, (right_border + left_border)/2, ball_j, params, tick) / delta_t), lowest, pocket_is_lowest, POCKET, ball_j);
	if(!ball_j->pocket_available){
		lowest = find_lowest_m(floor(corner_shortcut(ball_j, params, tick, left_border, top_border+l/sqrt(2)) / delta_t), lowest, pocket_is_lowest, CORNER, ball_j);
		lowest = find_lowest_m(floor(corner_shortcut(ball_j, params, tick, left_border+l/sqrt(2), top_border) / delta_t), lowest, pocket_is_lowest, CORNER, ball_j);
		lowest = find_lowest_m(floor(corner_shortcut(ball_j, params, tick, left_border, bottom_border-l/sqrt(2)) / delta_t), lowest, pocket_is_lowest, CORNER, ball_j);
		lowest = find_lowest_m(floor(corner_shortcut(ball_j, params, tick, left_border+l/sqrt(2), bottom_border) / delta_t), lowest, pocket_is_lowest, CORNER, ball_j);
		lowest = find_lowest_m(floor(corner_shortcut(ball_j, params, tick, (right_border + left_border)/2-l/2, bottom_border) / delta_t), lowest, pocket_is_lowest, CORNER, ball_j);
		lowest = find_lowest_m(floor(corner_shortcut(ball_j, params, tick, (right_border + left_border)/2+l/2, bottom_border) / delta_t), lowest, pocket_is_lowest, CORNER, ball_j);
		lowest = find_lowest_m(floor(corner_shortcut(ball_j, params, tick, right_border, bottom_border-l/sqrt(2)) / delta_t), lowest, pocket_is_lowest, CORNER, ball_j);
		lowest = find_lowest_m(floor(corner_shortcut(ball_j, params, tick, right_border-l/sqrt(2), bottom_border) / delta_t), lowest, pocket_is_lowest, CORNER, ball_j);
		lowest = find_lowest_m(floor(corner_shortcut(ball_j, params, tick, right_border, top_border+l/sqrt(2)) / delta_t), lowest, pocket_is_lowest, CORNER, ball_j);
		lowest = find_lowest_m(floor(corner_shortcut(ball_j, params, tick, right_border-l/sqrt(2), top_border) / delta_t), lowest, pocket_is_lowest, CORNER, ball_j);
		lowest = find_lowest_m(floor(corner_shortcut(ball_j, params, tick, (right_border + left_border)/2-l/2, top_border) / delta_t), lowest, pocket_is_lowest, CORNER, ball_j);
		lowest = find_lowest_m(floor(corner_shortcut(ball_j, params, tick, (right_border + left_border)/2+l/2, top_border) / delta_t), lowest, pocket_is_lowest, CORNER, ball_j);
		
		lowest = find_lowest_m(floor(border_shortcut(ball_j, params, ball_j->y, top_border+R, tick, 'v') / delta_t), lowest, pocket_is_lowest, BORDER, ball_j);
		lowest = find_lowest_m(floor(border_shortcut(ball_j, params, ball_j->y, bottom_border-R, tick, 'v') / delta_t), lowest, pocket_is_lowest, BORDER, ball_j);
		lowest = find_lowest_m(floor(border_shortcut(ball_j, params, ball_j->x, left_border+R, tick, 'h') / delta_t), lowest, pocket_is_lowest, BORDER, ball_j);
		lowest = find_lowest_m(floor(border_shortcut(ball_j, params, ball_j->x, right_border-R, tick, 'h') / delta_t), lowest, pocket_is_lowest, BORDER, ball_j);
	}
	return lowest;
}

double shortcut_step(struct Ball balls[], struct Params *params, double tick, enum State *state){
	double lowest = nan("1");
	bool pocket_is_lowest_a = false;
	bool pocket_is_lowest = false;
	for (int i = 0; i < params->n; ++i){
		double lowest0 = autonomical_lowest(&balls[i], params, tick, &pocket_is_lowest_a);
		lowest = find_lowest(lowest0, lowest, &pocket_is_lowest, pocket_is_lowest_a);		
	}
	for (int i = 0; i < params->n; ++i){
		for (int j = i+1; j < params->n; ++j){
			lowest = find_lowest_m(floor(balls_shortcut(params, balls[i].x, balls[i].y, balls[i].v_x*exp(-params->mu*(tick - balls[i].tick_base)), 
				balls[i].v_y*exp(-params->mu*(tick - balls[i].tick_base)), balls[j].x, balls[j].y, balls[j].v_x*exp(-params->mu*(tick - balls[j].tick_base)), 
				balls[j].v_y*exp(-params->mu*(tick - balls[j].tick_base))) / params->delta_t), lowest, &pocket_is_lowest, COLLISION, &balls[i]);	
		}
	}
	if(isinf(lowest)){
		*state = BALL_BEYOND_TABLE;
		visualize(balls, params, tick, tick+300000*params->delta_t, 0.01);
		return nan("1");
	}
	visualize(balls, params, tick, tick+lowest*params->delta_t, 0.1);
	if(pocket_is_lowest) *state = BALL_BEYOND_TABLE;
	if(!isnan(lowest)){
		for (int i = 0; i < params->n; ++i)
		{
			balls[i].x_0 = balls[i].x;
			balls[i].y_0 = balls[i].y;
			double dx = movement_integral(balls[i].v_x, params->mu, tick-balls[i].tick_base+lowest*params->delta_t, tick-balls[i].tick_base);
			double dy = movement_integral(balls[i].v_y, params->mu, tick-balls[i].tick_base+lowest*params->delta_t, tick-balls[i].tick_base);
			balls[i].x += dx;
			balls[i].y += dy;
		}
	}
	else{
		if(params->mu > 0){
			visualize(balls, params, tick, tick+300000*params->delta_t, 0.1);
			putting_out(balls, params->n, params->mu, tick);
		}
	}
	return lowest;
}

void mechanics_step(struct Ball balls[], struct Params *params, double tick, int *n_count, int *d_count){
	for (int i = 0; i < params->n; ++i)
	{
		balls[i].x_0 = balls[i].x;
		balls[i].y_0 = balls[i].y;
		double dx = movement_integral(balls[i].v_x, params->mu, tick-balls[i].tick_base+params->delta_t, tick-balls[i].tick_base);
		double dy = movement_integral(balls[i].v_y, params->mu, tick-balls[i].tick_base+params->delta_t, tick-balls[i].tick_base);
		balls[i].x += dx;
		balls[i].y += dy;
		double v = sqrt(balls[i].v_x*balls[i].v_x+balls[i].v_y*balls[i].v_y);
		if(balls[i].corner_available){
			if((balls[i].x_corner-balls[i].x)*(balls[i].x_corner-balls[i].x) + (balls[i].y_corner-balls[i].y)*(balls[i].y_corner-balls[i].y) < params->R*params->R){
				corner_return(&balls[i], params, tick);
			}
		}
		else if(balls[i].border_available){
			if(balls[i].y > params->bottom_border - params->R){
				horizontal_return(&balls[i], params, params->bottom_border - params->R, tick);
				(*n_count)++;
			}
			if(balls[i].y < params->top_border + params->R){
				horizontal_return(&balls[i], params, params->top_border + params->R, tick);
				(*n_count)++;
			}
			if(balls[i].x > params->right_border - params->R){
				vertical_return(&balls[i], params, params->right_border - params->R, tick);
				(*n_count)++;
			}
			if(balls[i].x < params->left_border + params->R){
				vertical_return(&balls[i], params, params->left_border + params->R, tick);
				(*n_count)++;
			}
		}
	}
	for (int i = 0; i < params->n; ++i){
		for (int j = i+1; j < params->n; ++j){
			if((balls[i].x-balls[j].x)*(balls[i].x-balls[j].x)+(balls[i].y-balls[j].y)*(balls[i].y-balls[j].y) <= 4*params->R*params->R && j != i){
				collision(&balls[i], &balls[j], params, tick);
				(*d_count)++;
			}
		}
	}
}

enum State check_table(struct Ball balls[], struct Params *params, double tick){
	for (int i = 0; i < params->n; ++i){
		if((balls[i].x < params->left_border) ||
			((balls[i].x >= params->left_border && balls[i].x < params->left_border + params->l/sqrt(2)) && 
				(balls[i].y < -balls[i].x + params->top_border+params->left_border+params->l/sqrt(2) || 
				balls[i].y > balls[i].x + params->bottom_border - params->left_border - params->l/sqrt(2))) || 
			((balls[i].x >= params->left_border + params->l/sqrt(2) && balls[i].x <= params->right_border - params->l/sqrt(2)) && 
				(balls[i].y < params->top_border || balls[i].y > params->bottom_border)) ||
			((balls[i].x > params->right_border - params->l/sqrt(2) && balls[i].x <= params->right_border) && 
				(balls[i].y > -balls[i].x + params->bottom_border + params->right_border - params->l/sqrt(2) || 
				balls[i].y < balls[i].x + params->top_border - params->right_border + params->l/sqrt(2))) || (balls[i].x > params->right_border)) 
			return BALL_BEYOND_TABLE;
	}
	return LACK_OF_ENERGY;
}

bool all_balls_in_move(struct Ball balls[], int n){
	bool all_in_move = true;
	for(int i=0; i<n; i++){
		all_in_move = balls[i].v_x != 0.0;
		if(!all_in_move){
			return false;
		}
	}
	return true;
}

void write_velocity_distribution(FILE *fp, struct Ball balls[], int n, int *write_counter){
	for(int i = 0; i < n; i++){
	   fprintf(fp, "%lf,", balls[i].v_x);
	}
	(*write_counter)++;
	fprintf(fp, "\n");
}

#endif