/*************************************************************************************
			       DEPARTMENT OF ELECTRICAL AND ELECTRONIC ENGINEERING
					   		     IMPERIAL COLLEGE LONDON 

 				      EE 3.19: Real Time Digital Signal Processing
					       Dr Paul Mitcheson and Daniel Harvey

				        		 PROJECT: Frame Processing

 				            ********* ENHANCE. C **********
							 Shell for speech enhancement 

  		Demonstrates overlap-add frame processing (interrupt driven) on the DSK. 

 *************************************************************************************
 				             By Danny Harvey: 21 July 2006
							 Updated for use on CCS v4 Sept 2010
 ************************************************************************************/
/*
 *	You should modify the code so that a speech enhancement project is built 
 *  on top of this template.
 */
/**************************** Pre-processor statements ******************************/
//  library required when using calloc
#include <stdlib.h>
//  Included so program can make use of DSP/BIOS configuration tool.  
#include "dsp_bios_cfg.h"
/* The file dsk6713.h must be included in every program that uses the BSL.  This 
   example also includes dsk6713_aic23.h because it uses the 
   AIC23 codec module (audio interface). */
#include "dsk6713.h"
#include "dsk6713_aic23.h"
#include <math.h>					/* math library (trig function*/
#include "cmplx.h"      			/* Some functions to help with Complex algebra and FFT. */
#include "fft_functions.h"  
#include <helper_functions_ISR.h>	// Some functions to help with writing/reading the audio ports when using interrupts.
/*****DEFINES*****/
#define WINCONST 0.85185			/* 0.46/0.54 for Hamming window */
#define FSAMP 8000.0				/* sample frequency, ensure this matches Config for AIC */
#define FFTLEN 256					/* fft length = frame length 256/8000 = 32 ms*/
#define NFREQ (1+FFTLEN/2)			/* number of frequency bins from a real FFT 129 */
#define OVERSAMP 4					/* oversampling ratio (2 or 4) */  
#define FRAMEINC (FFTLEN/OVERSAMP)	/* Frame increment 64 */
#define CIRCBUF (FFTLEN+FRAMEINC)	/* length of I/O buffers 320 */
#define OUTGAIN 16000.0				/* Output gain for DAC */
#define INGAIN  (1.0/16000.0)		/* Input gain for ADC  */
#define PI 3.141592653589793		/* PI defined here for use in your code */
#define TFRAME FRAMEINC/FSAMP       /* time between calculation of each frame 64/8000 = 8ms*/

/*************************Switches to Control Optimizations****************/
#define TIMELIMIT 312			/*number of frames to be compared before updating M_1(omega) TIMELIMIT = 312 = 2.5s/0.008s*/
float KSCALE = 0.0821;				/*for TAU 80ms*/
//#define KSCALE 0.7788  			/*for TAU 32ms very large attenuation*/  

#define enhLPF 1
//#define enhLPFPOWER 1
#define enhLPFNoise 1
#define overSub 1
int chooseThreshold = 1; /* 1->10*/
float alphamax = 1000;
int freqCap = 32;
int highFreqCap = 32;
/*************************End of Switches to Control Optimizations****************/
/******************************* Global declarations ********************************/
/* Audio port configuration settings: these values set registers in the AIC23 audio 
   interface to configure it. See TI doc SLWS106D 3-3 to 3-10 for more info. */
DSK6713_AIC23_Config Config = { \
			 /**********************************************************************/
			 /*   REGISTER	            FUNCTION			      SETTINGS         */ 
			 /**********************************************************************/\
    0x0017,  /* 0 LEFTINVOL  Left line input channel volume  0dB                   */\
    0x0017,  /* 1 RIGHTINVOL Right line input channel volume 0dB                   */\
    0x01f9,  /* 2 LEFTHPVOL  Left channel headphone volume   0dB                   */\
    0x01f9,  /* 3 RIGHTHPVOL Right channel headphone volume  0dB                   */\
    0x0011,  /* 4 ANAPATH    Analog audio path control       DAC on, Mic boost 20dB*/\
    0x0000,  /* 5 DIGPATH    Digital audio path control      All Filters off       */\
    0x0000,  /* 6 DPOWERDOWN Power down control              All Hardware on       */\
    0x0043,  /* 7 DIGIF      Digital audio interface format  16 bit                */\
    0x008d,  /* 8 SAMPLERATE Sample rate control        8 KHZ-ensure matches FSAMP */\
    0x0001   /* 9 DIGACT     Digital interface activation    On                    */\
			 /**********************************************************************/
};
  
DSK6713_AIC23_CodecHandle H_Codec;	/* Codec handle:- a variable used to identify audio interface */
float	*inbuffer, *outbuffer;   		/* Input/output circular buffers */
float	*inframe, *outframe;          /* Input and output frames */
float	*inwin, *outwin;              /* Input and output windows */
float	ingain, outgain;				/* ADC and DAC gains */ 
float	cpufrac; 						/* Fraction of CPU time used */
volatile int io_ptr=0;              /* Input/ouput pointer for circular buffers */
volatile int frame_ptr=0;           /* Frame pointer */
volatile int timePtr = 1;			/*range timePtr : {1,312}. noise estimation update every 2.5sconds*/
complex *cplxBuf, *outCplxBuf;
float	*mag1, *mag2, *mag3, *mag4, *gMag, *noiseMag, *thisMag, *yMag, *thisLPFMag, *noiseLPFMag;
float	floatMAX = 0x7FFFFFFF;
float	ALPHA = 20;
float	LAMBDA = 0.01;
int		idxFreq = 0;
 /******************************* Function prototypes *******************************/
void init_hardware(void);    		/* Initialize codec */ 
void init_HWI(void);            	/* Initialize hardware interrupts */
void init_buffers(void);			/* Initializes and zero fills buffers */
void init_window (void);			/* Initializes constants for Hamming window */
void ISR_AIC(void);             	/* Interrupt service routine for codec */
void process_frame(void);       	/* Frame processing routine */        
void floatToComplex (void);			/* Converts the input frame in float to complex number */
float square (float input);			/* takes square */
void findMagnitude (void);			/* Finds Magnitude of the complex number */
void LPFPower (void); 				/* computes the minimum input vector estimate based on LPF version of power */
void LPFX (void);					/* computes the minimum input vector estimate based on LPF version of magnitude */
void findMinLPF (void);				/* computes the minimum estimate based on LPF version of magnitude / power */
void noLPF (void);					/* computes the minimum based on magnitudes of input (NO LPF) */
void estimateNoiseX (void);			/* noise subtraction on non-LPF version */
void estimateNoiseLPF (void);		/* noise subtraction on LPF version */
void shiftMag (void);				/* shifts the minimum estimates */
void noiseSubtract (void);			/* performs noise subtraction by multiplying Y(w)=X(w)G(w)*/
void overSubtract(void);				/* performs oversubtract for lower frequency bins*/
void noiseThreshold1 (void);		/* performs thresholding based on lambda */
void noiseThreshold2 (void);		/* performs thresholding based on lambda */
void noiseThreshold3 (void);		/* performs thresholding based on lambda */
void noiseThreshold4 (void);		/* performs thresholding based on lambda */
void noiseThreshold5 (void);		/* performs thresholding based on lambda */
void noiseThreshold6 (void);		/* performs thresholding based on lambda */
void noiseThreshold7 (void);		/* performs thresholding based on lambda */
void noiseThreshold8 (void);		/* performs thresholding based on lambda */
void noiseThreshold9 (void);		/* performs thresholding based on lambda */
void noiseThreshold10 (void);		/* performs thresholding based on lambda */
void complexToFloat (void);			/* converts the complex time domain back to floating point */
/********************************** Main routine ************************************/
void main()
{
	init_buffers();					/* Initializes and zero fills buffers */
  	init_hardware();				/* initialize board and the audio port */
  	init_HWI();    					/* initialize hardware interrupts */  
  	init_window();					/* initialize algorithm constants */
  	ingain=INGAIN;
  	outgain=OUTGAIN;        					  
  	while(1) process_frame();		/* main loop, wait for interrupt */
}
/********************************** init_hardware() *********************************/  
void init_hardware()
{
    // Initialize the board support library, must be called first 
    DSK6713_init();
    // Start the AIC23 codec using the settings defined above in config 
    H_Codec = DSK6713_AIC23_openCodec(0, &Config);
	/* Function below sets the number of bits in word used by MSBSP (serial port) for 
	receives from AIC23 (audio port). We are using a 32 bit packet containing two 
	16 bit numbers hence 32BIT is set for  receive */
	MCBSP_FSETS(RCR1, RWDLEN1, 32BIT);	
	/* Configures interrupt to activate on each consecutive available 32 bits 
	from Audio port hence an interrupt is generated for each L & R sample pair */	
	MCBSP_FSETS(SPCR1, RINTM, FRM);
	/* These commands do the same thing as above but applied to data transfers to the 
	audio port */
	MCBSP_FSETS(XCR1, XWDLEN1, 32BIT);	
	MCBSP_FSETS(SPCR1, XINTM, FRM);	
}
/********************************** init_HWI() **************************************/ 
void init_HWI(void)
{
	IRQ_globalDisable();			// Globally disables interrupts
	IRQ_nmiEnable();				// Enables the NMI interrupt (used by the debugger)
	IRQ_map(IRQ_EVT_RINT1,4);		// Maps an event to a physical interrupt
	IRQ_enable(IRQ_EVT_RINT1);		// Enables the event
	IRQ_globalEnable();				// Globally enables interrupts

}
/******************************** process_frame() ***********************************/  
void process_frame(void)
{
	int k, m, io_ptr0;
	/* work out fraction of available CPU time used by algorithm */    
	cpufrac = ((float) (io_ptr & (FRAMEINC - 1)))/FRAMEINC;
	/* wait until io_ptr is at the start of the current frame */ 	
	while((io_ptr/FRAMEINC) != frame_ptr);	
	/* then increment the framecount (wrapping if required) */ 
	if (++frame_ptr >= (CIRCBUF/FRAMEINC)) frame_ptr=0; /*jump by 64, range is 0->5*/
 	/* save a pointer to the position in the I/O buffers (inbuffer/outbuffer) where the 
 	data should be read (inbuffer) and saved (outbuffer) for the purpose of processing */
 	io_ptr0=frame_ptr * FRAMEINC;
	/* copy input data from inbuffer into inframe (starting from the pointer position) */
	m = io_ptr0;
	/*windowing on input frame using Hanning Window*/  
    for (k=0; k < FFTLEN; k ++)
	{          
		inframe[k] = inbuffer[m] * inwin[k]; 
		if (++m >= CIRCBUF) m=0; /* wrap if required */
	} 	
	/************************* DO PROCESSING OF FRAME  HERE **************************/
	floatToComplex();
	fft(FFTLEN, cplxBuf);
	findMagnitude();
	/*start processing here*/
	/*process input frame such as LPF*/
	#ifdef enhLPF
		#ifdef enhLPFPOWER
			LPFPower();
		#else
			LPFX();
		#endif		
	#else
		noLPF();
	#endif
	/*noise estimation*/
	if(++timePtr<TIMELIMIT){}
	else{		
		#ifdef enhLPFNoise
			estimateNoiseLPF();
		#else
			estimateNoiseX();
		#endif
		shiftMag();
		timePtr = 1;
	}
	#ifdef overSub
		overSubtract();
	#endif	
	switch (chooseThreshold) {
		case 1: noiseThreshold1(); break;
		case 2: noiseThreshold2(); break;
		case 3: noiseThreshold3(); break;
		case 4: noiseThreshold4(); break;
		case 5: noiseThreshold5(); break;
		case 6: noiseThreshold6(); break;
		case 7: noiseThreshold7(); break;
		case 8: noiseThreshold8(); break;
		case 9: noiseThreshold9(); break;
		case 10: noiseThreshold10(); break;
	}
	
	/*noise subtraction*/	
	noiseSubtract();	
	/*END PROCESSING HERE*/	
	ifft(FFTLEN, outCplxBuf);
	complexToFloat();
	/********************************************************************************/
    /* multiply outframe by output window and overlap-add into output buffer */  
	m=io_ptr0;
    for (k=0;k<(FFTLEN-FRAMEINC);k++) 
	{    										/* this loop adds into outbuffer */                       
	  	outbuffer[m] = outbuffer[m]+outframe[k]*outwin[k];   
		if (++m >= CIRCBUF) m=0; /* wrap if required */
	}         
    for (;k<FFTLEN;k++) 
	{                           
		outbuffer[m] = outframe[k]*outwin[k];   /* this loop over-writes outbuffer */        
	    m++;
	}	                                   
}       

/*************************** INTERRUPT SERVICE ROUTINE  *****************************/   
void ISR_AIC(void)
{       
	// Map this to the appropriate interrupt in the TCF file
	short sample;
	/* Read and write the ADC and DAC using inbuffer and outbuffer */
	sample = mono_read_16Bit();
	inbuffer[io_ptr] = ((float)sample)*ingain;
	/* write new output data */
	mono_write_16Bit((int)(outbuffer[io_ptr]*outgain)); 
	/* update io_ptr and check for buffer wraparound */    
	if (++io_ptr >= CIRCBUF) io_ptr=0;
}
/************************************************************************************/

void init_buffers (void) {
	/*  Initialize and zero fill arrays */
	int k;
	inbuffer	= (float *) calloc(CIRCBUF, sizeof(float));	/* Input array */
    outbuffer	= (float *) calloc(CIRCBUF, sizeof(float));	/* Output array */
	inframe		= (float *) calloc(FFTLEN, sizeof(float));	/* Array for processing*/
    outframe	= (float *) calloc(FFTLEN, sizeof(float));	/* Array for processing*/
    inwin		= (float *) calloc(FFTLEN, sizeof(float));	/* Input window */
    outwin		= (float *) calloc(FFTLEN, sizeof(float));	/* Output window */
	cplxBuf		= (complex *) calloc(FFTLEN, sizeof(complex)); /* Array for processing*/
	outCplxBuf	= (complex *) calloc(FFTLEN, sizeof(complex)); /* Array for processing*/
	mag1			= (float *) calloc(FFTLEN, sizeof(float)); /* magnitude of FFT*/
	mag2			= (float *) calloc(FFTLEN, sizeof(float)); /* magnitude of FFT*/
	mag3			= (float *) calloc(FFTLEN, sizeof(float)); /* magnitude of FFT*/
	mag4			= (float *) calloc(FFTLEN, sizeof(float)); /* magnitude of FFT*/
	gMag			= (float *) calloc(FFTLEN, sizeof(float)); /* magnitude of FFT*/
	noiseMag		= (float *) calloc(FFTLEN, sizeof(float)); /* magnitude of FFT*/
	thisMag			= (float *) calloc(FFTLEN, sizeof(float)); /* magnitude of FFT*/
	yMag			= (float *) calloc(FFTLEN, sizeof(float)); /* magnitude of FFT*/
	thisLPFMag		= (float *) calloc(FFTLEN, sizeof(float)); /* magnitude of FFT*/
	noiseLPFMag		= (float *) calloc(FFTLEN, sizeof(float)); /* magnitude of FFT*/
	for(k = 0; k < FFTLEN; k++) {
		mag1[k] = floatMAX;
		mag2[k] = floatMAX;
		mag3[k] = floatMAX;
		mag4[k] = floatMAX;
	}
}

void init_window (void) {
	int k;
	for (k=0;k<FFTLEN;k++)
	{                           
	inwin[k] = sqrt((1.0-WINCONST*cos(PI*(2*k+1)/FFTLEN))/OVERSAMP);
	outwin[k] = inwin[k]; 
	} 
}

void floatToComplex (void){
	for (idxFreq = 0; idxFreq < FFTLEN; idxFreq++ ){
		cplxBuf[idxFreq].r = inframe[idxFreq];
		cplxBuf[idxFreq].i = 0;
	} 
}

void findMagnitude (void) {
	for (idxFreq = 0; idxFreq < FFTLEN; idxFreq++){
		thisMag[idxFreq] = cabs(cplxBuf[idxFreq]);
	}
}

void LPFPower (void) {
	ALPHA = 2;
	for (idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		thisLPFMag[idxFreq] = sqrt((1-KSCALE)*square(thisMag[idxFreq])+KSCALE*(thisLPFMag[idxFreq]));
	}
	findMinLPF();
}

void LPFX (void) {
	for (idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
			thisLPFMag[idxFreq] = (1-KSCALE)*thisMag[idxFreq]+KSCALE*thisLPFMag[idxFreq];
		}
	findMinLPF();
}

void findMinLPF (void) {
	for (idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		if(thisLPFMag[idxFreq] < mag1[idxFreq]){
			mag1[idxFreq]=thisLPFMag[idxFreq];
		}
	}
}

void noLPF (void) {
	for (idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		if(thisMag[idxFreq] < mag1[idxFreq]){
			mag1[idxFreq]=thisMag[idxFreq];
		}
	}
}

void estimateNoiseLPF(void) {
	float minMag;
	for(idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){/*0->256*/
		minMag = mag1[idxFreq]; /* initialize with the a magnitude from M1*/
		if(minMag > mag2[idxFreq]){	minMag = mag2[idxFreq];}
		if(minMag > mag3[idxFreq]){	minMag = mag3[idxFreq];}
		if(minMag > mag4[idxFreq]){	minMag = mag4[idxFreq];}
		noiseLPFMag[idxFreq] = ALPHA * minMag; //this is the N(omega)
	}
	for(idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){/*0->256*/
		noiseMag[idxFreq] = (1-KSCALE)*noiseLPFMag[idxFreq]+KSCALE*noiseMag[idxFreq]; ;//this is the Pt(omega)
	}
}

void estimateNoiseX(void) {
	float minMag;
	for(idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){/*0->256*/
		minMag = mag1[idxFreq]; /* initialize with the a magnitude from M1*/
		if(minMag > mag2[idxFreq]){	minMag = mag2[idxFreq];}
		if(minMag > mag3[idxFreq]){	minMag = mag3[idxFreq];}
		if(minMag > mag4[idxFreq]){	minMag = mag4[idxFreq];}
		noiseMag[idxFreq] = ALPHA * minMag;
	}
}

void overSubtract (void){
	/*oversubtract only for low frequency bins*/
	for(idxFreq = FFTLEN ; idxFreq > FFTLEN-highFreqCap ; idxFreq --){/*0->256*/
		noiseMag[idxFreq] = noiseMag[idxFreq]*alphamax;
	}	
}

void shiftMag(void) {
	int k;
	/*shift the estimate vectors and resets M1 to floatmax.*/
	mag4=mag3;
	mag3=mag2;
	mag2=mag1;
	for(k = 0; k < FFTLEN; k++) { 
		mag1[k] = floatMAX;
	}
}

void noiseSubtract(void) {
	/*noise subtraction*/
	for(idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		outCplxBuf[idxFreq] = rmul(gMag[idxFreq], cplxBuf[idxFreq]);
	}
}

void noiseThreshold1 (void) {
	float minMag;
	for(idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		minMag = 1 - noiseMag[idxFreq]/thisMag[idxFreq];
		if(minMag <= LAMBDA){
			gMag[idxFreq] = LAMBDA;
		}
		else{
			gMag[idxFreq] = minMag;
		}
	}
}

void noiseThreshold2 (void) {
	float minMag;
	float threshold;
	for(idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		threshold = LAMBDA*noiseMag[idxFreq]/thisMag[idxFreq];
		minMag = 1 - noiseMag[idxFreq]/thisMag[idxFreq];
		if(minMag <= threshold){
			gMag[idxFreq] = threshold;
		}
		else{
			gMag[idxFreq] = minMag;
		}
	}
}

void noiseThreshold3 (void) {
	float minMag;
	float threshold;
	for(idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		threshold = LAMBDA*thisLPFMag[idxFreq]/thisMag[idxFreq];
		minMag = 1 - noiseMag[idxFreq]/thisMag[idxFreq];
		if(minMag <= threshold){
			gMag[idxFreq] = threshold;
		}
		else{
			gMag[idxFreq] = minMag;
		}
	}
}

void noiseThreshold4 (void) {
	float minMag;
	float threshold;
	for(idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		threshold = LAMBDA*noiseMag[idxFreq]/thisLPFMag[idxFreq];
		minMag = 1 - noiseMag[idxFreq]/thisLPFMag[idxFreq];
		if(minMag <= threshold){
			gMag[idxFreq] = threshold;
		}
		else{
			gMag[idxFreq] = minMag;
		}
	}
}

void noiseThreshold5 (void) {
	float minMag;
	for(idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		minMag = 1 - noiseMag[idxFreq]/thisLPFMag[idxFreq];
		if(minMag <= LAMBDA){
			gMag[idxFreq] = LAMBDA;
		}
		else{
			gMag[idxFreq] = minMag;
		}
	}
}

void noiseThreshold6 (void) {
	float minMag;
	for(idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		minMag = sqrt(1 - square(noiseMag[idxFreq])/square(thisMag[idxFreq]));
		if(minMag <= LAMBDA){
			gMag[idxFreq] = LAMBDA;
		}
		else{
			gMag[idxFreq] = minMag;
		}
	}
}

void noiseThreshold7 (void) {
	float minMag;
	float threshold;
	for(idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		threshold = LAMBDA*noiseMag[idxFreq]/thisMag[idxFreq];
		minMag = sqrt(1 - square(noiseMag[idxFreq])/square(thisMag[idxFreq]));
		if(minMag <= threshold){
			gMag[idxFreq] = threshold;
		}
		else{
			gMag[idxFreq] = minMag;
		}
	}
}

void noiseThreshold8 (void) {
	float minMag;
	float threshold;
	for(idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		threshold = LAMBDA*thisLPFMag[idxFreq]/thisMag[idxFreq];
		minMag = sqrt(1 - square(noiseMag[idxFreq])/square(thisMag[idxFreq]));
		if(minMag <= threshold){
			gMag[idxFreq] = threshold;
		}
		else{
			gMag[idxFreq] = minMag;
		}
	}
}

void noiseThreshold9 (void) {
	float minMag;
	float threshold;
	for(idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		threshold = LAMBDA*noiseMag[idxFreq]/thisLPFMag[idxFreq];
		minMag = sqrt(1 - square(noiseMag[idxFreq])/square(thisLPFMag[idxFreq]));
		if(minMag <= threshold){
			gMag[idxFreq] = threshold;
		}
		else{
			gMag[idxFreq] = minMag;
		}
	}
}

void noiseThreshold10 (void) {
	float minMag;
	for(idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		minMag = sqrt(1 - square(noiseMag[idxFreq])/square(thisLPFMag[idxFreq]));
		if(minMag <= LAMBDA){
			gMag[idxFreq] = LAMBDA;
		}
		else{
			gMag[idxFreq] = minMag;
		}
	}
}

void complexToFloat (void) {
	for (idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		outframe[idxFreq] = outCplxBuf[idxFreq].r;		
	}
}

float square (float input) {
	return input*input;
}
