#ifndef HEADER_FILE
#define HEADER_FILE


#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <complex.h>
	
double eps=1e-12;
typedef enum {PROPORTIONAL_TO_VELOCITY, UNIFORMLY_DECELERATED} Motion_mode;

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
	Motion_mode motion_mode;
};

enum State {BALL_BEYOND_TABLE, LACK_OF_ENERGY, GAME_ON, ERROR};
enum Return_mode {POCKET, CORNER, BORDER, COLLISION};

unsigned int solveP3(double *x, double a, double b, double c) {
	double PI = 3.141592653589793238463L;
	double M_2PI = 2*PI;
	double a2 = a*a;
    double q  = (a2 - 3*b)/9;
	double r  = (a*(2*a2-9*b) + 27*c)/54;
    double r2 = r*r;
	double q3 = q*q*q;
	double A,B;
    	if(r2<q3) 
    	{
    		double t=r/sqrt(q3);
    		if( t<-1) t=-1;
    		if( t> 1) t= 1;
    		t=acos(t);
    		a/=3; q=-2*sqrt(q);
    		x[0]=q*cos(t/3)-a;
    		x[1]=q*cos((t+M_2PI)/3)-a;
    		x[2]=q*cos((t-M_2PI)/3)-a;
    		return 3;
    	} 
    	else 
    	{
    		A =-pow(fabs(r)+sqrt(r2-q3),1./3);
    		if( r<0 ) A=-A;
    		B = (0==A ? 0 : q/A);
          
		a/=3;
		x[0] =(A+B)-a;
		x[1] =-0.5*(A+B)-a;
		x[2] = 0.5*sqrt(3.)*(A-B);
		if(fabs(x[2])<eps) { x[2]=x[1]; return 2; }
		
		return 1;
        }
}

//---------------------------------------------------------------------------
// returns roots of quartic polynomial x^4 + a*x^3 + b*x^2 + c*x + d
double solve_quartic(double a, double b, double c, double d)
{
	double a3 = -b;
	double b3 =  a*c -4.*d;
	double c3 = -a*a*d - c*c + 4.*b*d;

	// cubic resolvent
	// y^3 − b*y^2 + (ac−4d)*y − a^2*d−c^2+4*b*d = 0
	
	double x3[3];
	unsigned int iZeroes = solveP3(x3, a3, b3, c3);
	
	double q1, q2, p1, p2, D, sqD, y;

	y = x3[0];
	// The essence - choosing Y with maximal absolute value.
	if(iZeroes != 1)
	{
		if(fabs(x3[1]) > fabs(y)) y = x3[1];
		if(fabs(x3[2]) > fabs(y)) y = x3[2];
	}

	// h1+h2 = y && h1*h2 = d  <=>  h^2 -y*h + d = 0    (h === q)

	D = y*y - 4*d;
	if(fabs(D) < eps) //in other words - D==0
	{
		q1 = q2 = y * 0.5;
		// g1+g2 = a && g1+g2 = b-y   <=>   g^2 - a*g + b-y = 0    (p === g)
		D = a*a - 4*(b-y);
		if(fabs(D) < eps) //in other words - D==0
			p1 = p2 = a * 0.5;

		else
		{
			sqD = sqrt(D);
			p1 = (a + sqD) * 0.5;
			p2 = (a - sqD) * 0.5;
		}
	}
	else
	{
		sqD = sqrt(D);
		q1 = (y + sqD) * 0.5;
		q2 = (y - sqD) * 0.5;
		// g1+g2 = a && g1*h2 + g2*h1 = c       ( && g === p )  Krammer
		p1 = (a*q1-c)/(q1-q2);
		p2 = (c-a*q2)/(q1-q2);
	}

	double complex retval[4];

	// solving quadratic eq. - x^2 + p1*x + q1 = 0
	D = p1*p1 - 4*q1;
	if(D < 0.0)
	{
		retval[0] = -p1 * 0.5 + sqrt(-D) * 0.5 * I;
		retval[1] = conj(retval[0]);
	}
	else
	{
		sqD = sqrt(D);
		retval[0] = (-p1 + sqD) * 0.5;
		retval[1] = (-p1 - sqD) * 0.5;
	}
	D = p2*p2 - 4*q2;
	if(D < 0.0)
	{
		retval[2] = -p2 * 0.5 + sqrt(-D) * 0.5 * I;
		retval[3] = conj(retval[2]);
	}
	else
	{
		sqD = sqrt(D);
		retval[2] = (-p2 + sqD) * 0.5;
		retval[3] = (-p2 - sqD) * 0.5;
	}
    double real_solution;
    if(cimag(retval[3]) != 0.0) real_solution = nan("1");
    else real_solution = (double)creal(retval[3]);
    return real_solution;
}

//---------------------------------------------------------------------------
// returns roots of quadratic polynomial ax^2 + bx + c
double solve_quadratic(double a, double b, double c){
	double delta = b*b - 4*a*c;
	double x1 = (-b + sqrt(delta))/2/a;
	double x2 = (-b - sqrt(delta))/2/a;
	if(x1 < x2) return x1;
	else return x2;
}

double sgn(double v) {
  if (v < 0) return -1;
  if (v > 0) return 1;
  return 0;
}

double sinus(double x, double y){
	if(x == 0 && y == 0) return 0;
	else return y / sqrt(x*x + y*y);
}

double cosinus(double x, double y){
	if(x == 0 && y == 0) return 0;
	else return x / sqrt(x*x + y*y);
}

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
	printf("%lf\n", theta);
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
		printf("%lf\n", balls[i].x);
		printf("%lf\n", balls[i].y);
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

double movement_integral(double v_0, double mu, double upper_limes, double lower_limes, Motion_mode motion_mode){
	if(motion_mode == UNIFORMLY_DECELERATED){
		//printf("v0=%lf, mu=%lf\n", v_0, mu);
		//printf("s = %lf - %lf\n", v_0*(upper_limes - lower_limes), mu/2*(upper_limes - lower_limes)*(upper_limes - lower_limes));
		return v_0*(upper_limes - lower_limes) - mu/2*(upper_limes - lower_limes)*(upper_limes - lower_limes);
	}
	if(motion_mode == PROPORTIONAL_TO_VELOCITY){
		if(mu == 0.0) return v_0 * (upper_limes - lower_limes);
		else return v_0/mu*(exp(-mu*lower_limes) - exp(-mu*upper_limes));
	}
}

void horizontal_return(struct Ball *ball_j, struct Params *params, double border, double tick){
	//printf("horizontal_return\n");
	double c;
	double v_x, v_y;
	double mu_x, mu_y;
	if(params->motion_mode == UNIFORMLY_DECELERATED){
		double sin_a = sinus(ball_j->v_x, ball_j->v_y);
		double cos_a = cosinus(ball_j->v_x, ball_j->v_y);
		v_x = ball_j->v_x-cos_a*params->mu*cos_a*(tick - ball_j->tick_base);
		v_y = ball_j->v_y-sin_a*params->mu*sin_a*(tick - ball_j->tick_base);
		c = solve_quadratic(-params->mu*sin_a/2, v_y, ball_j->y_0 - border);
		mu_x = params->mu * cos_a;
		mu_y = params->mu * sin_a;
	}
	if(params->motion_mode == PROPORTIONAL_TO_VELOCITY){
		if(params->mu == 0.0) c = (border-ball_j->y_0) / ball_j->v_y; 
		else c = 1/params->mu * log(1/(1-params->mu*(border-ball_j->y_0)/(ball_j->v_y)*exp(params->mu*(tick-ball_j->tick_base))));
		v_x = ball_j->v_x;
		v_y = ball_j->v_y;
		mu_x = params->mu;
		mu_y = params->mu;
	}
	if(c < 0) return;
	double x_z = ball_j->x_0 + movement_integral(v_x, mu_x, tick-ball_j->tick_base+c, tick-ball_j->tick_base, params->motion_mode);
	if(x_z > (params->left_border + params->l/sqrt(2)) && x_z < (params->right_border - params->l/sqrt(2))){
		ball_j->y = ball_j->y_0 + movement_integral(v_y, mu_y, tick-ball_j->tick_base+c, tick-ball_j->tick_base, params->motion_mode);
		ball_j->v_y *= -1;
		v_y *= -1;
		if(params->motion_mode == UNIFORMLY_DECELERATED) mu_y *= -1;
		ball_j->y = ball_j->y + movement_integral(v_y, mu_y, tick-ball_j->tick_base+params->delta_t, tick-ball_j->tick_base+c, params->motion_mode);
		ball_j->x = ball_j->x_0 + movement_integral(v_x, mu_x, tick-ball_j->tick_base+params->delta_t, tick-ball_j->tick_base, params->motion_mode);
	}
	else{
		ball_j->y = ball_j->y_0 + movement_integral(v_y, mu_y, tick-ball_j->tick_base+params->delta_t, tick-ball_j->tick_base, params->motion_mode);
		ball_j->x = ball_j->x_0 + movement_integral(v_x, mu_x, tick-ball_j->tick_base+params->delta_t, tick-ball_j->tick_base, params->motion_mode);
	}
}

void vertical_return(struct Ball *ball_j, struct Params *params, double border, double tick){
	//printf("vertical_return\n");
	double c;
	double v_x, v_y;
	double mu_x, mu_y;
	if(params->motion_mode == UNIFORMLY_DECELERATED){
		double sin_a = sinus(ball_j->v_x, ball_j->v_y);
		double cos_a = cosinus(ball_j->v_x, ball_j->v_y);
		v_x = ball_j->v_x-cos_a*params->mu*cos_a*(tick - ball_j->tick_base);
		v_y = ball_j->v_y-sin_a*params->mu*sin_a*(tick - ball_j->tick_base);
		c = solve_quadratic(-params->mu*cos_a/2, v_x, ball_j->x_0 - border);
		mu_x = params->mu * cos_a;
		mu_y = params->mu * sin_a;
	}
	if(params->motion_mode == PROPORTIONAL_TO_VELOCITY){
		if(params->mu == 0.0) c = (border-ball_j->x_0) / ball_j->v_x; 
		else c = 1/params->mu * log(1/(1-params->mu*(border-ball_j->x_0)/ball_j->v_x*exp(params->mu*(tick-ball_j->tick_base))));
		v_x = ball_j->v_x;
		v_y = ball_j->v_y;
		mu_x = params->mu;
		mu_y = params->mu;
	}
	if(c < 0) return;
	double y_z = ball_j->y_0 + movement_integral(v_y, mu_y, tick-ball_j->tick_base+c, tick-ball_j->tick_base, params->motion_mode);
	if(y_z > (params->top_border + params->l/sqrt(2)) && y_z < (params->bottom_border - params->l/sqrt(2))){
		ball_j->x = ball_j->x_0 + movement_integral(v_x, mu_x, tick-ball_j->tick_base+c, tick-ball_j->tick_base, params->motion_mode);
		ball_j->v_x *= -1;
		v_x *= -1;
		if(params->motion_mode == UNIFORMLY_DECELERATED) mu_x *= -1;
		ball_j->y = ball_j->y_0 + movement_integral(v_y, mu_y, tick-ball_j->tick_base+params->delta_t, tick-ball_j->tick_base, params->motion_mode);
		ball_j->x = ball_j->x + movement_integral(v_x, mu_x, tick-ball_j->tick_base+params->delta_t, tick-ball_j->tick_base+c, params->motion_mode);
	}
	else{
		ball_j->y = ball_j->y_0 + movement_integral(v_y, mu_y, tick-ball_j->tick_base+params->delta_t, tick-ball_j->tick_base, params->motion_mode);
		ball_j->x = ball_j->x_0 + movement_integral(v_x, mu_x, tick-ball_j->tick_base+params->delta_t, tick-ball_j->tick_base, params->motion_mode);
	}
}

void corner_return(struct Ball *ball_j, struct Params *params, double tick){ // ta funkcja kurwa nadal nie odbija tam gdzie trzeba!!!!
	//printf("corner_return\n");
	double c;
	double v_x, v_y;
	double mu_x, mu_y;
	double sin_a = sinus(ball_j->v_x, ball_j->v_y);
	double cos_a = cosinus(ball_j->v_x, ball_j->v_y);
	if(params->motion_mode == UNIFORMLY_DECELERATED){
		double x_p = ball_j->x_0 - ball_j->x_corner;
		double y_p = ball_j->y_0 - ball_j->y_corner;
		mu_x = params->mu * cos_a;
		mu_y = params->mu * sin_a;
		v_x = ball_j->v_x - params->mu*cos_a*(tick - ball_j->tick_base);
		v_y = ball_j->v_y - params->mu*sin_a*(tick - ball_j->tick_base);
		double A = params->mu*params->mu/4;
		double B = -params->mu*(v_x*cos_a + v_y*sin_a);
		double C = v_x*v_x + v_y*v_y - params->mu * (x_p*cos_a + y_p*sin_a);
		double D = 2*(x_p*v_x + y_p*v_y);
		double E = x_p*x_p + y_p*y_p - params->R*params->R;
		B /= A;
		C /= A;
		D /= A;
		E /= A;
		c = solve_quartic(B, C, D, E);
	}
	else{
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
		mu_x = params->mu;
		mu_y = params->mu;
		v_x = ball_j->v_x;
		v_y = ball_j->v_y;
	}
	ball_j->x = ball_j->x_0 + movement_integral(v_x, mu_x, tick-ball_j->tick_base+c, tick-ball_j->tick_base, params->motion_mode);
	ball_j->y = ball_j->y_0 + movement_integral(v_y, mu_y, tick-ball_j->tick_base+c, tick-ball_j->tick_base, params->motion_mode);
	double v_xp, v_yp;
	if(params->motion_mode == UNIFORMLY_DECELERATED){
		v_xp = ball_j->v_x - cos_a*params->mu*(c + tick - ball_j->tick_base);
		v_yp = ball_j->v_y - sin_a*params->mu*(c + tick - ball_j->tick_base);
	}
	else{
		v_xp = ball_j->v_x*exp(-params->mu*(tick - ball_j->tick_base + c));
		v_yp = ball_j->v_y*exp(-params->mu*(tick - ball_j->tick_base + c));
	}
	//printf("v_xp=%lf, v_yp=%lf\n", v_xp, v_yp);
	//printf("ball=(%lf, %lf), corner=(%lf, %lf)\n", ball_j->x, ball_j->y, ball_j->x_corner, ball_j->y_corner);
	double v = sqrt(v_xp*v_xp+v_yp*v_yp);
	double alpha = acos((ball_j->x_corner - ball_j->x)/params->R);
	if((ball_j->y_corner - ball_j->y)/params->R < 0) alpha = 2*M_PI - alpha;
	double beta = acos(v_xp/v);
	if(v_yp/v < 0) beta = 2*M_PI - beta;
	//printf("v=%lf, alpha=%lf, beta=%lf\n", v, alpha, beta);
	//printf("mój tan=%lf, moja alpha=%lf\n", (ball_j->y_corner - ball_j->y)/(ball_j->x_corner - ball_j->x), atan((ball_j->y_corner - ball_j->y)/(ball_j->x_corner - ball_j->x)));
	double vq = v * cos(beta - alpha);
	double vp = v * sin(beta - alpha);
	ball_j->v_x = vq * cos(alpha + M_PI) + vp * cos(alpha + M_PI/2);
	ball_j->v_y = vq * sin(alpha + M_PI) + vp * sin(alpha + M_PI/2);
	//printf("v_q=%lf, v_p=%lf\n", vq, vp);
	//printf("v_x=%lf, v_y=%lf\n", ball_j->v_x, ball_j->v_y);
	ball_j->tick_base = tick + c;
	if(params->motion_mode == UNIFORMLY_DECELERATED){
		double sin_a = sinus(ball_j->v_x, ball_j->v_y);
		double cos_a = cosinus(ball_j->v_x, ball_j->v_y);
		mu_x = params->mu * cos_a;
		mu_y = params->mu * sin_a;	
	}
	v_x = ball_j->v_x;
	v_y = ball_j->v_y;
	ball_j->x += movement_integral(v_x, mu_x, params->delta_t-c, 0, params->motion_mode);
	ball_j->y += movement_integral(v_y, mu_y, params->delta_t-c, 0, params->motion_mode);
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
	double v_oxdr, v_oydr;
	double v_xj, v_xk, v_yj, v_yk;
	double mu_xj, mu_xk, mu_yj, mu_yk;
	double sin_a, cos_a, sin_b, cos_b;
	double c;
	if(params->motion_mode == UNIFORMLY_DECELERATED){
		cos_a = cosinus(ball_j->v_x, ball_j->v_y);
		sin_a = sinus(ball_j->v_x, ball_j->v_y);
		cos_b = cosinus(ball_k->v_x, ball_k->v_y);
		sin_b = sinus(ball_k->v_x, ball_k->v_y);
		v_xj = ball_j->v_x - cos_a*params->mu*(tick - ball_j->tick_base);
		v_xk = ball_k->v_x - cos_b*params->mu*(tick - ball_k->tick_base);
		v_yj = ball_j->v_y - sin_a*params->mu*(tick - ball_j->tick_base);
		v_yk = ball_k->v_y - sin_b*params->mu*(tick - ball_k->tick_base);
		v_oxdr = ball_j->v_x - cos_a*params->mu*(tick - ball_j->tick_base) - ball_k->v_x + cos_b*params->mu*(tick - ball_k->tick_base);
		v_oydr = ball_j->v_y - sin_a*params->mu*(tick - ball_j->tick_base) - ball_k->v_y + sin_b*params->mu*(tick - ball_k->tick_base);
		mu_xj = params->mu*cos_a;
		mu_xk = params->mu*cos_b;
		mu_yj = params->mu*sin_a;
		mu_yk = params->mu*sin_b;

		double u_dx = params->mu/2*(cos_b - cos_a);
		double u_dy = params->mu/2*(sin_b - sin_a);
		double A = u_dx*u_dx + u_dy*u_dy;
		double B = 2*(u_dx*v_oxdr + u_dy*v_oydr);
		double C = v_oxdr*v_oxdr + v_oydr*v_oydr + 2*u_dx*x_0dr + 2*u_dy*y_0dr;
		double D = 2*(x_0dr*v_oxdr + y_0dr*v_oydr);
		double E = x_0dr*x_0dr + y_0dr*y_0dr - 4*params->R*params->R;
		B /= A;
		C /= A;
		D /= A;
		E /= A;
		c = solve_quartic(B, C, D, E);
	}
	else{
		v_xj = ball_j->v_x;
		v_xk = ball_k->v_x;
		v_yj = ball_j->v_y;
		v_yk = ball_k->v_y;
		v_oxdr = ball_j->v_x*exp(-mu*(tick-ball_j->tick_base)) - ball_k->v_x*exp(-mu*(tick-ball_k->tick_base));
		v_oydr = ball_j->v_y*exp(-mu*(tick-ball_j->tick_base)) - ball_k->v_y*exp(-mu*(tick-ball_k->tick_base));
		mu_xj = params->mu;
		mu_xk = params->mu;
		mu_yj = params->mu;
		mu_yk = params->mu;
		double gamma = v_oxdr*v_oxdr + v_oydr*v_oydr;
		double beta = 2*(v_oxdr*x_0dr+v_oydr*y_0dr);
		double alpha = x_0dr*x_0dr + y_0dr*y_0dr - 4*R*R;
		// 4)
		double sqrt_delta = sqrt(beta*beta - 4*gamma*alpha);
		// 5)
		if(mu == 0.0){
			c = -(beta + sqrt_delta) / 2 / gamma;
		}
		else{
			double w = -(beta + sqrt_delta) / 2 / gamma;
			c = log(1/(1-mu*w)) / mu;
		}
	}
	// 3)
	
	// 7)

	double delta_x_cdj = movement_integral(v_xj, mu_xj, tick-ball_j->tick_base+c, tick-ball_j->tick_base, params->motion_mode);
	double delta_y_cdj = movement_integral(v_yj, mu_yj, tick-ball_j->tick_base+c, tick-ball_j->tick_base, params->motion_mode);
	double delta_x_cdk = movement_integral(v_xk, mu_xk, tick-ball_k->tick_base+c, tick-ball_k->tick_base, params->motion_mode);
	double delta_y_cdk = movement_integral(v_yk, mu_yk, tick-ball_k->tick_base+c, tick-ball_k->tick_base, params->motion_mode);
	// 8)
	double xdj = ball_j->x_0 + delta_x_cdj;
	double ydj = ball_j->y_0 + delta_y_cdj;
	double xdk = ball_k->x_0 + delta_x_cdk;
	double ydk = ball_k->y_0 + delta_y_cdk;
	// 9)
	double sin_phi = (xdk - xdj) / sqrt((xdk - xdj)*(xdk - xdj) + (ydk - ydj)*(ydk - ydj));
	double cos_phi = (ydj - ydk) / sqrt((xdk - xdj)*(xdk - xdj) + (ydk - ydj)*(ydk - ydj));

	double v_psidj, v_etadj, v_psidk, v_etadk;
	if(params->motion_mode == UNIFORMLY_DECELERATED){
		v_psidj = (v_xj - mu_xj*c)*cos_phi + (v_yj - mu_yj*c)*sin_phi;
		v_etadj = (v_yj - mu_yj*c)*cos_phi - (v_xj - mu_xj*c)*sin_phi;
		v_psidk = (v_xk - mu_xk*c)*cos_phi + (v_yk - mu_yk*c)*sin_phi;
		v_etadk = (v_yk - mu_yk*c)*cos_phi - (v_xk - mu_xk*c)*sin_phi;
	}
	else{
		v_psidj = (ball_j->v_x*cos_phi + ball_j->v_y*sin_phi)*exp(-mu*(tick-ball_j->tick_base+c));
		v_etadj = (ball_j->v_y*cos_phi - ball_j->v_x*sin_phi)*exp(-mu*(tick-ball_j->tick_base+c));
		v_psidk = (ball_k->v_x*cos_phi + ball_k->v_y*sin_phi)*exp(-mu*(tick-ball_k->tick_base+c));
		v_etadk = (ball_k->v_y*cos_phi - ball_k->v_x*sin_phi)*exp(-mu*(tick-ball_k->tick_base+c));
	}
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

	double w_psidj, w_psidk;
	if(params->motion_mode == UNIFORMLY_DECELERATED){
		w_psidj = sqrt((v_xj - mu_xj*c)*(v_xj - mu_xj*c) + (v_yj - mu_yj*c)*(v_yj - mu_yj*c))*(sin_thetadj*sin_phi+cos_phi*cos_thetadj);
		w_psidk = sqrt((v_xk - mu_xk*c)*(v_xk - mu_xk*c) + (v_yk - mu_yk*c)*(v_yk - mu_yk*c))*(sin_thetadk*sin_phi+cos_phi*cos_thetadk);
	}
	else{
		w_psidj = sqrt(ball_j->v_x*ball_j->v_x + ball_j->v_y*ball_j->v_y)*exp(-mu*(tick-ball_j->tick_base+c))*(sin_thetadj*sin_phi+cos_phi*cos_thetadj);
		w_psidk = sqrt(ball_k->v_x*ball_k->v_x + ball_k->v_y*ball_k->v_y)*exp(-mu*(tick-ball_k->tick_base+c))*(sin_thetadk*sin_phi+cos_phi*cos_thetadk);
	}
	//////////////////////////
	ball_j->v_x = w_psidj*cos_phi - w_etadj*sin_phi;
	ball_j->v_y = w_psidj*sin_phi + w_etadj*cos_phi;
	ball_k->v_x = w_psidk*cos_phi - w_etadk*sin_phi;
	ball_k->v_y = w_psidk*sin_phi + w_etadk*cos_phi;

	if(params->motion_mode == UNIFORMLY_DECELERATED){
		cos_a = cosinus(ball_j->v_x, ball_j->v_y);
		sin_a = sinus(ball_j->v_x, ball_j->v_y);
		cos_b = cosinus(ball_k->v_x, ball_k->v_y);
		sin_b = sinus(ball_k->v_x, ball_k->v_y);
		v_xj = ball_j->v_x;
		v_xk = ball_k->v_x;
		v_yj = ball_j->v_y;
		v_yk = ball_k->v_y;
		mu_xj = params->mu*cos_a;
		mu_xk = params->mu*cos_b;
		mu_yj = params->mu*sin_a;
		mu_yk = params->mu*sin_b;
	}
	else{
		v_xj = ball_j->v_x;
		v_xk = ball_k->v_x;
		v_yj = ball_j->v_y;
		v_yk = ball_k->v_y;
	}
	// 10)
	ball_j->tick_base = tick + c;
	ball_k->tick_base = tick + c;
	// 11)
	ball_j->x = xdj + movement_integral(v_xj, mu_xj, delta_t-c, 0, params->motion_mode);
	ball_j->y = ydj + movement_integral(v_yj, mu_yj, delta_t-c, 0, params->motion_mode);
	ball_k->x = xdk + movement_integral(v_xk, mu_xk, delta_t-c, 0, params->motion_mode);
	ball_k->y = ydk + movement_integral(v_yk, mu_yk, delta_t-c, 0, params->motion_mode);
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

double ud_putting_out(struct Ball balls[], struct Params *params, int putting_out_id, double tick){
	double cos_ap = balls[putting_out_id].v_x / sqrt(balls[putting_out_id].v_x*balls[putting_out_id].v_x + balls[putting_out_id].v_y*balls[putting_out_id].v_y);
	double v_xp = balls[putting_out_id].v_x - params->mu*cos_ap*(tick - balls[putting_out_id].tick_base);
	double t = v_xp/(params->mu*cos_ap);
	visualize(balls, params, tick, tick+t, 0.1);
	for (int i = 0; i < params->n; ++i){
		balls[i].x_0 = balls[i].x;
		balls[i].y_0 = balls[i].y;
		double dx, dy;
		double v_x, v_y;
		double mu_x, mu_y;
		double sin_a = sinus(balls[i].v_x, balls[i].v_y);
		double cos_a = cosinus(balls[i].v_x, balls[i].v_y);
		mu_x = params->mu*cos_a;
		mu_y = params->mu*sin_a;
		v_x = balls[i].v_x - params->mu*cos_a*(tick - balls[i].tick_base);
		v_y = balls[i].v_y - params->mu*sin_a*(tick - balls[i].tick_base);
		dx = movement_integral(v_x, mu_x, tick-balls[i].tick_base+t, tick-balls[i].tick_base, params->motion_mode);
		dy = movement_integral(v_y, mu_y, tick-balls[i].tick_base+t, tick-balls[i].tick_base, params->motion_mode);
		balls[i].x += dx;
		balls[i].y += dy;
	}
	balls[putting_out_id].v_x = 0.0;
	balls[putting_out_id].v_y = 0.0;
	return t;
}

double border_shortcut(struct Ball *ball_j, struct Params *params, double q0, double border, double tick, char mode){
	if(ball_j->v_x == 0.0 && ball_j->v_y == 0.0) return nan("1");
	double v0q;
	double func;
	double v_x, v_y;
	double mu_x, mu_y;
	if(params->motion_mode == UNIFORMLY_DECELERATED){
		double sin_a = sinus(ball_j->v_x, ball_j->v_y);
		double cos_a = cosinus(ball_j->v_x, ball_j->v_y);
		v_x = ball_j->v_x - params->mu*cos_a*(tick - ball_j->tick_base);
		v_y = ball_j->v_y - params->mu*sin_a*(tick - ball_j->tick_base);
		if(mode == 'v'){
			v0q = ball_j->v_y - sin_a*params->mu*(tick - ball_j->tick_base);
			func = sin_a;
		}
		else{
			v0q = ball_j->v_x - cos_a*params->mu*(tick - ball_j->tick_base);
			func = cos_a;
		}
		mu_x = params->mu*cos_a;
		mu_y = params->mu*sin_a;
	}
	else{
		if(mode == 'v') v0q = ball_j->v_y*exp(-params->mu*(tick - ball_j->tick_base));
		else v0q = ball_j->v_x*exp(-params->mu*(tick - ball_j->tick_base));
		v_x = ball_j->v_x*exp(-params->mu*(tick - ball_j->tick_base));
		v_y = ball_j->v_y*exp(-params->mu*(tick - ball_j->tick_base));
		mu_x = params->mu;
		mu_y = params->mu;
	}
	double t;
	if(params->motion_mode == UNIFORMLY_DECELERATED){
		t = solve_quadratic(-func/2*params->mu, v0q, q0 - border);
	}
	else{
		if(params->mu == 0.0) t = (border - q0) / v0q;
		else t = -1/params->mu*log(1 - params->mu/v0q*(border - q0));
	}
	if(mode == 'v'){
		double x1;
		x1 = ball_j->x + movement_integral(v_x, mu_x, tick-ball_j->tick_base+t, tick-ball_j->tick_base, params->motion_mode);
		if(x1 < params->left_border + params->l/sqrt(2) || (x1 > (params->right_border + params->left_border)/2-params->l/2 && 
			x1 < (params->right_border + params->left_border)/2+params->l/2) || x1 > params->right_border - params->l/sqrt(2)) return nan("1");
	}
	else{
		double y1;
		y1 = ball_j->y + movement_integral(v_y, mu_y, tick-ball_j->tick_base+t, tick-ball_j->tick_base, params->motion_mode);
		if(y1 < params->top_border + params->l/sqrt(2) || y1 > params->bottom_border - params->l/sqrt(2)) return nan("1");
	}
	if(t >= 0){
		//printf("q0=%lf, border=%lf, v_x=%lf, v_y=%lf\n", q0, border, v_x, v_y);
		//printf("t=%d, b=%d, l=%d, r=%d\n", params->top_border, params->bottom_border, params->left_border, params->right_border);
		ball_j->border_available = true;
		double v_xp = ball_j->v_x - mu_x*(tick - ball_j->tick_base);
		double tp = v_xp/(mu_x);
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
	double x_p, y_p;
	double v_x, v_y;
	double mu_x, mu_y;
	double sin_a = sinus(ball_j->v_x, ball_j->v_y);
	double cos_a = cosinus(ball_j->v_x, ball_j->v_y);
	if(params->motion_mode == UNIFORMLY_DECELERATED){
		mu_x = params->mu*cos_a;
		mu_y = params->mu*sin_a;
		v_x = ball_j->v_x - params->mu*cos_a*(tick - ball_j->tick_base);
		v_y = ball_j->v_y - params->mu*sin_a*(tick - ball_j->tick_base);
	}
	else{
		mu_x = params->mu;
		mu_y = params->mu;
		v_x = ball_j->v_x;
		v_y = ball_j->v_y;
	}
	x_p = ball_j->x + movement_integral(v_x, mu_x, tick-ball_j->tick_base+t, tick-ball_j->tick_base, params->motion_mode);
	y_p = ball_j->y + movement_integral(v_y, mu_y, tick-ball_j->tick_base+t, tick-ball_j->tick_base, params->motion_mode);
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
	if(ball_j->v_x == 0.0 && ball_j->v_y == 0.0) return nan("1");
	double t;
	if(params->motion_mode == UNIFORMLY_DECELERATED){
		double v_x, v_y;
		double sin_a = sinus(ball_j->v_x, ball_j->v_y);
		double cos_a = cosinus(ball_j->v_x, ball_j->v_y);
		double x_p = ball_j->x - x_co;
		double y_p = ball_j->y - y_co;
		v_x = ball_j->v_x - params->mu*cos_a*(tick - ball_j->tick_base);
		v_y = ball_j->v_y - params->mu*sin_a*(tick - ball_j->tick_base);
		double A = params->mu*params->mu/4;
		double B = -params->mu*(v_x*cos_a + v_y*sin_a);
		double C = v_x*v_x + v_y*v_y - params->mu * (x_p*cos_a + y_p*sin_a);
		double D = 2*(x_p*v_x + y_p*v_y);
		double E = x_p*x_p + y_p*y_p - params->R*params->R;
		B /= A;
		C /= A;
		D /= A;
		E /= A;
		t = solve_quartic(B, C, D, E);	
	}
	else{
		if(params->mu == 0.0){
			double x_p = ball_j->x - x_co;
			double y_p = ball_j->y - y_co;
			double A = ball_j->v_x*ball_j->v_x + ball_j->v_y*ball_j->v_y;
			double B = 2*(ball_j->v_x*x_p + ball_j->v_y*y_p);
			double C = x_p*x_p + y_p*y_p - params->R*params->R;
			double delta = B*B - 4*A*C;
			if(delta < 0) return nan("1");
			double t = (-B - sqrt(delta))/2/A;
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
			t = -1/params->mu*log(g);
		}
	}
	bool border_crossing = check_border_crossing(ball_j, params, tick, x_co, y_co, t);
	if(!border_crossing && !isnan(t) && t > 0){
		ball_j->corner_available = true;
		ball_j->x_corner = x_co;
		ball_j->y_corner = y_co;
	}
	else return nan("1");
	return t;
}


double balls_shortcut(struct Ball *ball_i, struct Ball *ball_j, struct Params *params, double tick){
	double x_10, y_10, v_1x, v_1y, x_20, y_20, v_2x, v_2y;
	double sin_a, cos_a, sin_b, cos_b;
	double t;
	x_10 = ball_i->x;
	x_20 = ball_j->x;
	y_10 = ball_i->y;
	y_20 = ball_j->y;
	if(params->motion_mode == UNIFORMLY_DECELERATED){
		cos_a = cosinus(ball_i->v_x, ball_i->v_y);
		sin_a = sinus(ball_i->v_x, ball_i->v_y);
		cos_b = cosinus(ball_j->v_x, ball_j->v_y);
		sin_b = sinus(ball_j->v_x, ball_j->v_y);
		v_1x = ball_i->v_x - cos_a*params->mu*(tick - ball_i->tick_base);
		v_1y = ball_i->v_y - sin_a*params->mu*(tick - ball_i->tick_base);
		v_2x = ball_j->v_x - cos_b*params->mu*(tick - ball_j->tick_base);
		v_2y = ball_j->v_y - sin_b*params->mu*(tick - ball_j->tick_base);
	}
	else{
		v_1x = ball_i->v_x*exp(-params->mu*(tick - ball_i->tick_base));
		v_1y = ball_i->v_y*exp(-params->mu*(tick - ball_i->tick_base));
		v_2x = ball_j->v_x*exp(-params->mu*(tick - ball_j->tick_base));
		v_2y = ball_j->v_y*exp(-params->mu*(tick - ball_j->tick_base));
	}
	if(v_1x == 0.0 && v_1y == 0.0 && v_2x == 0.0 && v_2y == 0.0) return nan("1");
	double dx = x_10 - x_20;
	double dy = y_10 - y_20;
	if(params->motion_mode == UNIFORMLY_DECELERATED){
		double u_dx = params->mu/2*(cos_b - cos_a);
		double u_dy = params->mu/2*(sin_b - sin_a);
		double dvx = v_1x - v_2x;
		double dvy = v_1y - v_2y;
		double A = u_dx*u_dx + u_dy*u_dy;
		double B = 2*(u_dx*dvx + u_dy*dvy);
		double C = dvx*dvx + dvy*dvy + 2*u_dx*dx + 2*u_dy*dy;
		double D = 2*(dx*dvx + dy*dvy);
		double E = dx*dx + dy*dy - 4*params->R*params->R;
		B /= A;
		C /= A;
		D /= A;
		E /= A;
		t = solve_quartic(B, C, D, E);
	}
	else{
		if(params->mu == 0.0){
			double dvx = v_1x - v_2x;
			double dvy = v_1y - v_2y;
			double A = dvx*dvx + dvy*dvy;
			double B = 2*(dx*dvx + dy*dvy);
			double C = dx*dx + dy*dy - 4*params->R*params->R;
			double delta = B*B - 4*A*C;
			if(delta < 0) return nan("1");
			t = (-B - sqrt(delta))/2/A;
			
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
			t = -log(-(B-sqrt(delta))/2/A)/params->mu;
		}
	}
	if(t >= 0){
		return t;
	}
	else return nan("1");
}


double pocket_shortcut(double g, double b, double xc, struct Ball *ball_j, struct Params *params, double tick){
	if(ball_j->v_x == 0.0 && ball_j->v_y == 0.0) return nan("1");
	double v_x, v_y, tg;
	double sin_a = sinus(ball_j->v_x, ball_j->v_y);
	double cos_a = cosinus(ball_j->v_x, ball_j->v_y);
	if(params->motion_mode == UNIFORMLY_DECELERATED){
		v_x = ball_j->v_x - cos_a*params->mu*(tick - ball_j->tick_base);
		v_y = ball_j->v_y - sin_a*params->mu*(tick - ball_j->tick_base);
		tg = ball_j->v_y / ball_j->v_x;
	}
	else{
		v_x = ball_j->v_x*exp(-params->mu*(tick - ball_j->tick_base));
		v_y = ball_j->v_y*exp(-params->mu*(tick - ball_j->tick_base));
		tg = v_y / v_x;
	}
	double x3 = (ball_j->y-tg*ball_j->x - b) / (g - tg);
	double x1 = (ball_j->y-params->R*sqrt(1+tg*tg)-tg*ball_j->x - b) / (g - tg);
	double x2 = (ball_j->y+params->R*sqrt(1+tg*tg)-tg*ball_j->x - b) / (g - tg);
	
	if(((1+g*g)*(x1 - xc)*(x1 - xc) <= params->l*params->l/4) && ((1+g*g)*(x2 - xc)*(x2 - xc) <= params->l*params->l/4)){
		double t;
		if(params->motion_mode == UNIFORMLY_DECELERATED)
			t = solve_quadratic(-cos_a/2*params->mu, v_x, ball_j->x - x3);
		else{
			if(params->mu == 0.0) t = (x3 - ball_j->x) / v_x;
			else t = -log(1 - params->mu*(x3 - ball_j->x)/(v_x))/params->mu;
		}
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
	//printf("pocket = %lf\n", lowest);
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
		//printf("corner = %lf\n", lowest);
		lowest = find_lowest_m(floor(border_shortcut(ball_j, params, ball_j->y, top_border+R, tick, 'v') / delta_t), lowest, pocket_is_lowest, BORDER, ball_j);
		lowest = find_lowest_m(floor(border_shortcut(ball_j, params, ball_j->y, bottom_border-R, tick, 'v') / delta_t), lowest, pocket_is_lowest, BORDER, ball_j);
		lowest = find_lowest_m(floor(border_shortcut(ball_j, params, ball_j->x, left_border+R, tick, 'h') / delta_t), lowest, pocket_is_lowest, BORDER, ball_j);
		lowest = find_lowest_m(floor(border_shortcut(ball_j, params, ball_j->x, right_border-R, tick, 'h') / delta_t), lowest, pocket_is_lowest, BORDER, ball_j);
		//printf("border = %lf\n", lowest);
	}
	return lowest;
}

double shortcut_step(struct Ball balls[], struct Params *params, double tick, enum State *state, int *putting_out_id){
	double lowest = nan("1");
	bool pocket_is_lowest_a = false;
	bool pocket_is_lowest = false;
	for (int i = 0; i < params->n; ++i){
		double lowest0 = autonomical_lowest(&balls[i], params, tick, &pocket_is_lowest_a);
		lowest = find_lowest(lowest0, lowest, &pocket_is_lowest, pocket_is_lowest_a);		
	}
	for (int i = 0; i < params->n; ++i){
		for (int j = i+1; j < params->n; ++j){
			lowest = find_lowest_m(floor(balls_shortcut(&balls[i], &balls[j], params, tick) / params->delta_t), lowest, &pocket_is_lowest, COLLISION, &balls[i]);	
		}
	}
	if(isinf(lowest)){
		*state = BALL_BEYOND_TABLE;
		visualize(balls, params, tick, tick+300000*params->delta_t, 0.01);
		return nan("1");
	}
	if(pocket_is_lowest) *state = BALL_BEYOND_TABLE;
	double min_putting_time = -1;
	if(params->motion_mode == UNIFORMLY_DECELERATED){
		for (int i = 0; i < params->n; ++i){
			double cos_a = balls[i].v_x / sqrt(balls[i].v_x*balls[i].v_x + balls[i].v_y*balls[i].v_y);
			double v_x = balls[i].v_x - params->mu*cos_a*(tick - balls[i].tick_base);
			if(is_lower(floor(v_x/(params->mu*cos_a)/params->delta_t), lowest)){
				if(min_putting_time == -1){
					min_putting_time = v_x/(params->mu*cos_a);
					*putting_out_id = i;
				}
				else{
					if(min_putting_time > v_x/(params->mu*cos_a)){
						min_putting_time = v_x/(params->mu*cos_a);
						*putting_out_id = i;
					}
				}
			}
		}
	}
	if(*putting_out_id != -1) return 0;
	if(!isnan(lowest)){
		visualize(balls, params, tick, tick+lowest*params->delta_t, 0.1);
		for (int i = 0; i < params->n; ++i)
		{
			balls[i].x_0 = balls[i].x;
			balls[i].y_0 = balls[i].y;
			double dx, dy;
			double v_x, v_y;
			double mu_x, mu_y;
			if(params->motion_mode == UNIFORMLY_DECELERATED){
				double sin_a = sinus(balls[i].v_x, balls[i].v_y);
				double cos_a = cosinus(balls[i].v_x, balls[i].v_y);
				mu_x = params->mu*cos_a;
				mu_y = params->mu*sin_a;
				v_x = balls[i].v_x - params->mu*cos_a*(tick - balls[i].tick_base);
				v_y = balls[i].v_y - params->mu*sin_a*(tick - balls[i].tick_base);
			}
			else{
				mu_x = params->mu;
				mu_y = params->mu;
				v_x = balls[i].v_x;
				v_y = balls[i].v_y;
			}
			dx = movement_integral(v_x, mu_x, tick-balls[i].tick_base+lowest*params->delta_t, tick-balls[i].tick_base, params->motion_mode);
			dy = movement_integral(v_y, mu_y, tick-balls[i].tick_base+lowest*params->delta_t, tick-balls[i].tick_base, params->motion_mode);
			balls[i].x += dx;
			balls[i].y += dy;
		}
	}
	else{
		if(params->mu > 0 && params->motion_mode == PROPORTIONAL_TO_VELOCITY){
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
		double dx, dy;
		double v_x, v_y;
		double mu_x, mu_y;
		if(params->motion_mode == UNIFORMLY_DECELERATED){
			double sin_a = sinus(balls[i].v_x, balls[i].v_y);
			double cos_a = cosinus(balls[i].v_x, balls[i].v_y);
			mu_x = params->mu*cos_a;
			mu_y = params->mu*sin_a;
			v_x = balls[i].v_x - params->mu*cos_a*(tick - balls[i].tick_base);
			v_y = balls[i].v_y - params->mu*sin_a*(tick - balls[i].tick_base);
		}
		else{
			mu_x = params->mu;
			mu_y = params->mu;
			v_x = balls[i].v_x;
			v_y = balls[i].v_y;
		}
		dx = movement_integral(v_x, mu_x, tick-balls[i].tick_base+params->delta_t, tick-balls[i].tick_base, params->motion_mode);
		dy = movement_integral(v_y, mu_y, tick-balls[i].tick_base+params->delta_t, tick-balls[i].tick_base, params->motion_mode);
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
		//printf("x=%lf, y=%lf\n", balls[i].x, balls[i].y);
		//printf("l=%d, r=%d, t=%d, b=%d\n", params->left_border, params->right_border, params->top_border, params->bottom_border);
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