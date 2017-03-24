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
#include <math.h>					/* Math library (trig function) */
#include "cmplx.h"      			/* Some functions to help with Complex algebra and FFT. */
#include "fft_functions.h"  
#include <helper_functions_ISR.h>	/* Some functions to help with writing/reading the audio ports when using interrupts. */
#define WINCONST 0.85185			/* 0.46/0.54 for Hamming window */
#define FSAMP 8000.0				/* sample frequency, ensure this matches Config for AIC */
#define NFREQ (1+FFTLEN/2)			/* number of frequency bins from a real FFT 129 */
#define OVERSAMP 4					/* oversampling ratio (2 or 4) */  
#define FRAMEINC (FFTLEN/OVERSAMP)	/* Frame increment 64 */
#define CIRCBUF (FFTLEN+FRAMEINC)	/* length of I/O buffers 320 */
#define OUTGAIN 16000.0				/* Output gain for DAC */
#define INGAIN  (1.0/16000.0)		/* Input gain for ADC  */
#define PI 3.141592653589793		/* PI defined here for use in your code */
#define TFRAME FRAMEINC/FSAMP       /* time between calculation of each frame 64/8000 = 8ms*/

/*************************Switches to Control Optimizations****************/
#define enhLPF 1					/* Task 1: Filters input */
//#define enhLPFPOWER 1				/* Task 2: Filters based on power of input */
#define enhLPFNoise 1				/* Task 3: Filters the noise estimate */
int		chooseThreshold = 6; 		/* Task 4: 1->5. Task5 5: 6->10*/
#define overSub 1					/* Task 6: Performs oversubtraction for lower frequency bins */
#define FFTLEN 256					/* Task 7: FFT length = frame length 256/8000 = 32 ms*/
#define delayOutput 1				/* Task 8: Estimates based on adjacent frames */
float	TIMELIMIT = 312;			/* Task 9: Change number of frames to be compared before updating M_1(omega)*/
float	alphamax = 1000;			/* Used for oversubtraction in Task 6*/
int		freqCap = 5;				/* Used for oversubtraction in Task 6*/
float	ALPHA = 20;					/* Used to compensate for underestimation of noise, ALPHA*N(w)*/
float	LAMBDA = 0.001;				/* Used to threshold the lowest value of each bin, max{LAMBDA, 1-N(w)/X(w)*/
float	musicalThreshold = 5;  		/* Used as threshold for estimates based on adjacent frames N(w)/X(w) */
float	noiseThreshold = 1;  		/* Used as threshold for noise */
float	killThreshold = 0.1;		/* Removes content below killThreshold to combat musical noise */
float	KSCALE =  0.2019;			/* KSCALE = exp(-T/tau). For 200Hz cutoff, tau = 0.005, T = 8ms, KSCALE = exp(-8/5)*/

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
float	*inbuffer, *outbuffer;   	/* Input/output circular buffers */
float	*inframe, *outframe;        /* Input and output frames */
float	*inwin, *outwin;            /* Input and output windows */
float	ingain, outgain;			/* ADC and DAC gains */ 
float	cpufrac; 					/* Fraction of CPU time used */
volatile int io_ptr=0;              /* Input/ouput pointer for circular buffers */
volatile int frame_ptr=0;           /* Frame pointer */
volatile int timePtr = 1;			/* Range timePtr : {1,312}. noise estimation update every 2.5sconds*/
int		idxFreq = 0;				/* Global counter variable to avoid repetitive declaration*/
float	floatMAX = 0x7FFFFFFF;		/* Maximum float value used to initialize buffers */
float   ratio = 0;					/* Ratio of |N|/|X|. ratio explicitly declared for Watch window*/

/**************** Declare complex and float buffer pointers****************/
complex *cplxBuf;					/* Complex buffer to store input frames */
complex *outCplxBuf;				/* Complex buffer to store output frames, Y_t(w)*/
complex *outDelay1;					/* Task8: Complex buffer to store previous output frames, Y_t-1(w)*/
complex *outDelay2;					/* Task8: Complex buffer to store previous output frames, Y_t-2(w)*/
float	*mag1, *mag2, *mag3, *mag4;	/* Buffers to store minimum estimates of noise */
float   *gMag;						/* Buffer to store G(w), the subtraction threshold */
float   *noiseMag;					/* Buffer to store N(w), the magnitude of noise */
float   *thisMag;					/* Buffer to store X(w), the magnitude of input signal*/
float   *yMag;						/* Buffer to store Y(w), the magnitude of filtered signal*/
float   *thisLPFMag;				/* Buffer to store P(w), the magnitude of low-pass filtered input signal*/
float   *noiseLPFMag;				/* Buffer to store P_N(w), the magnitude of low-pass filtered noise signal*/
float   *outDelay1Ratio;			/* Task 8: Buffer to store ratio N(w)/X(w) for delayed output by 1 */
float	*outDelay1X;

 /******************************* Function prototypes *******************************/
void init_hardware(void);    		/* Initialize codec */ 
void init_HWI(void);            	/* Initialize hardware interrupts */
void init_buffers(void);			/* Initializes and zero fills buffers */
void init_window (void);			/* Initializes constants for Hamming window */
void ISR_AIC(void);             	/* Interrupt service routine for codec */
void process_frame(void);       	/* Frame processing routine */        
void floatToComplex (void);			/* Converts the input frame in float to complex number */
void findMagnitude (void);			/* Converts float to of the complex number, takes FFT, and finds magnitude using cabs()*/
void LPFPower (void); 				/* Use low-pass filtered version of |P(w)| to estimate noise */
void LPFX (void);					/* Use low-pass filtered version of |X(w)| to estimate noise */
void findMinLPF (void);				/* Computes the minimum estimate based on LPF version of magnitude / power */
void noLPF (void);					/* Computes the minimum based on magnitudes of input directly without LPF */
void estimateNoiseX (void);			/* Noise subtraction on non-LPF version */
void estimateNoiseLPF (void);		/* Noise subtraction on LPF version of noise */
void shiftMag (void);				/* Shift the noise estimate vectors and resets M1 to floatMAX.*/
void noiseSubtract (void);			/* Performs noise subtraction by multiplying Y(w) = X(w)G(w)*/
void overSubtract(void);			/* Performs oversubtract for lower frequency bins*/
void noiseThreshold1 (void);		/* Performs thresholding based on max{LAMBDA, 1-|N|/|X|}*/
void noiseThreshold2 (void);		/* Performs thresholding based on max{LAMBDA*|N|/|X|, 1-|N|/|X|} */
void noiseThreshold3 (void);		/* Performs thresholding based on max{LAMBDA*|P|/|X|, 1-|N|/|X|} */
void noiseThreshold4 (void);		/* Performs thresholding based on max{LAMBDA*|N|/|P|, 1-|N|/|P|} */
void noiseThreshold5 (void);		/* Performs thresholding based on max{LAMBDA, 1-|N|/|P|} */
void noiseThreshold6 (void);		/* Performs thresholding based on power: max{LAMBDA, sqrt(1-(|N|/|X|)^2} */
void noiseThreshold7 (void);		/* Performs thresholding based on power: max{LAMBDA*|N|/|X|, sqrt(1-(|N|/|X|)^2)} */
void noiseThreshold8 (void);		/* Performs thresholding based on power: max{LAMBDA*|P|/|X|, sqrt(1-(|N|/|X|)^2)} */
void noiseThreshold9 (void);		/* Performs thresholding based on power: max{LAMBDA*|N|/|P|, sqrt(1-(|N|/|P|)^2)} */
void noiseThreshold10 (void);		/* Performs thresholding based on power: max{LAMBDA, sqrt(1-(|N|/|P|)^2)} */
void complexToFloat (void);			/* Converts the complex time domain back to floating point */
void complexToFloatDelayed (void);	/* Task 8: Converts the complex time domain back to floating point */
float square (float input);			/* Returns squared value of floats */
complex minOfThree (complex in1, complex in2, complex in3); /*returns minimum of three complex data type inputs */
/********************************** Main routine ************************************/
void main()
{
	init_buffers();					/* Initializes and zero fills buffers */
  	init_hardware();				/* Initialize board and the audio port */
  	init_HWI();    					/* Initialize hardware interrupts */  
  	init_window();					/* Initialize algorithm constants */
  	ingain=INGAIN;					/* ADC Gain, 16000. */
  	outgain=OUTGAIN;				/* DAC Gain, 16000. */
  	while(1) process_frame();		/* Main loop, wait for interrupt */
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
	findMagnitude();		/* Converts float to of the complex number, takes FFT, and finds magnitude using cabs() */
	/*start processing here*/
	#ifdef enhLPF
		#ifdef enhLPFPOWER
			LPFPower();		/* Use low-pass filtered version of |P(w)| to estimate noise */
		#else
			LPFX();			/* Use low-pass filtered version of |X(w)| to estimate noise */
		#endif		
	#else
		noLPF();			/* Computes the minimum based on magnitudes of input directly without LPF */
	#endif
	/*noise estimation*/
	if(++timePtr<TIMELIMIT){}	/* TIMELIMIT chooses period (2.5s) to find estimate of noise*/
	else{		
		
		#ifdef enhLPFNoise
			estimateNoiseLPF();	/* Noise subtraction on LPF version of noise */
		#else
			estimateNoiseX();	/* Noise subtraction on non-LPF version */
		#endif
		shiftMag();		/* Shift the noise estimate vectors and resets M1 to floatMAX.*/
		timePtr = 1;
	}
	#ifdef overSub
		overSubtract();	/* Performs oversubtract for lower frequency bins*/
	#endif
	
	switch (chooseThreshold) {
		case 1:		noiseThreshold1(); break;	// Performs thresholding based on max{LAMBDA, 1-|N|/|X|}
		case 2:		noiseThreshold2(); break;	// Performs thresholding based on max{LAMBDA*|N|/|X|, 1-|N|/|X|}
		case 3:		noiseThreshold3(); break;	// Performs thresholding based on max{LAMBDA*|P|/|X|, 1-|N|/|X|}
		case 4:		noiseThreshold4(); break;	// Performs thresholding based on max{LAMBDA*|N|/|P|, 1-|N|/|P|}
		case 5:		noiseThreshold5(); break;	// Performs thresholding based on max{LAMBDA, 1-|N|/|P|}
		case 6:		noiseThreshold6(); break;	// Performs thresholding based on power: max{LAMBDA, sqrt(1-(|N|/|X|)^2} 
		case 7:		noiseThreshold7(); break;	// Performs thresholding based on power: max{LAMBDA*|N|/|X|, sqrt(1-(|N|/|X|)^2)}
		case 8:		noiseThreshold8(); break;	// Performs thresholding based on power: max{LAMBDA*|P|/|X|, sqrt(1-(|N|/|X|)^2)}
		case 9:		noiseThreshold9(); break;	// Performs thresholding based on power: max{LAMBDA*|N|/|P|, sqrt(1-(|N|/|P|)^2)}
		case 10:	noiseThreshold10(); break;	// Performs thresholding based on power: max{LAMBDA, sqrt(1-(|N|/|P|)^2)}
	}
	
	
	noiseSubtract();				/* Performs noise subtraction by multiplying Y(w) = X(w)G(w)*/	

	#ifdef delayOutput
		complexToFloatDelayed();	/* Task 8: Converts the complex time domain back to floating point */
	#else
		complexToFloat();			/* Converts the complex time domain back to floating point */
	#endif
	/*END PROCESSING HERE*/	
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
	inbuffer		= (float *) calloc(CIRCBUF, sizeof(float));	/* Input array */
    outbuffer		= (float *) calloc(CIRCBUF, sizeof(float));	/* Output array */
	inframe			= (float *) calloc(FFTLEN, sizeof(float));	/* Array for processing*/
    outframe		= (float *) calloc(FFTLEN, sizeof(float));	/* Array for processing*/
    inwin			= (float *) calloc(FFTLEN, sizeof(float));	/* Input window */
    outwin			= (float *) calloc(FFTLEN, sizeof(float));	/* Output window */
	cplxBuf			= (complex *) calloc(FFTLEN, sizeof(complex)); /* Array for processing*/
	outCplxBuf		= (complex *) calloc(FFTLEN, sizeof(complex)); /* Array for processing*/
	outDelay1 		= (complex *) calloc(FFTLEN, sizeof(complex)); /* Array for processing*/
	outDelay2 		= (complex *) calloc(FFTLEN, sizeof(complex)); /* Array for processing*/
	mag1			= (float *) calloc(FFTLEN, sizeof(float)); /* Buffers to store minimum estimates of noise */
	mag2			= (float *) calloc(FFTLEN, sizeof(float)); /* Buffers to store minimum estimates of noise */
	mag3			= (float *) calloc(FFTLEN, sizeof(float)); /* Buffers to store minimum estimates of noise */
	mag4			= (float *) calloc(FFTLEN, sizeof(float)); /* Buffers to store minimum estimates of noise */
	gMag			= (float *) calloc(FFTLEN, sizeof(float)); /* Buffer to store G(w), the subtraction threshold */
	noiseMag		= (float *) calloc(FFTLEN, sizeof(float)); /* Buffer to store N(w), the magnitude of noise */
	thisMag			= (float *) calloc(FFTLEN, sizeof(float)); /* Buffer to store X(w), the magnitude of input signal*/
	yMag			= (float *) calloc(FFTLEN, sizeof(float)); /* Buffer to store Y(w), the magnitude of filtered signal*/
	thisLPFMag		= (float *) calloc(FFTLEN, sizeof(float)); /* Buffer to store P(w), the magnitude of low-pass filtered input signal*/
	noiseLPFMag		= (float *) calloc(FFTLEN, sizeof(float)); /* Buffer to store P_N(w), the magnitude of low-pass filtered noise signal*/
	outDelay1Ratio  = (float *) calloc(FFTLEN, sizeof(float)); /* Task 8: Buffer to store ratio N(w)/X(w) for delayed output by 1 */
	outDelay1X		= (float *) calloc(FFTLEN, sizeof(float)); /* Task 8: Buffer to store X(w) for delayed output by 1 */
}

void init_window (void) {
	/* Initializes constants for Hamming window */
	int k;
	for (k=0;k<FFTLEN;k++)
	{                           
	inwin[k] = sqrt((1.0-WINCONST*cos(PI*(2*k+1)/FFTLEN))/OVERSAMP);
	outwin[k] = inwin[k]; 
	} 
}

float square (float input) {
	/* Returns squared value of floats */
	return input*input;
}

complex minOfThree (complex in1, complex in2, complex in3) {
	/*returns minimum of three complex data type inputs */
	complex complexLocal = in1;
	if ( cabs(in2) < cabs(complexLocal)){
		complexLocal = in2;
	}
	if ( cabs(in3) < cabs(complexLocal)){
		complexLocal = in3;
	}
	return complexLocal;
}

void floatToComplex (void){
	/* Converts the input frame in float to complex number */
	for (idxFreq = 0; idxFreq < FFTLEN; idxFreq++ ){
		cplxBuf[idxFreq].r = inframe[idxFreq];
		cplxBuf[idxFreq].i = 0;
	} 
}

void findMagnitude (void) {
	/* Converts float to of the complex number, takes FFT, and finds magnitude using cabs() */
	floatToComplex();		/* Converts the input frame in float to complex number */
	fft(FFTLEN, cplxBuf);
	/* Finds magnitude of the complex number using cabs() */
	for (idxFreq = 0; idxFreq < FFTLEN; idxFreq++){
		thisMag[idxFreq] = cabs(cplxBuf[idxFreq]);
	}
}

void LPFPower (void) {
	/* Use low-pass filtered version of |P(w)| to estimate noise */
	for (idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		thisLPFMag[idxFreq] = sqrt((1-KSCALE)*square(thisMag[idxFreq])+KSCALE*(thisLPFMag[idxFreq]));
	}
	findMinLPF();
}

void LPFX (void) {
	/* Use low-pass filtered version of |X(w)| to estimate noise */
	for (idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
			thisLPFMag[idxFreq] = (1-KSCALE)*thisMag[idxFreq]+KSCALE*thisLPFMag[idxFreq];
		}
	findMinLPF();
}

void findMinLPF (void) {
	/* Computes the minimum estimate based on LPF version of magnitude / power */
	for (idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		if(thisLPFMag[idxFreq] < mag1[idxFreq]){
			mag1[idxFreq]=thisLPFMag[idxFreq];
		}
	}
}

void noLPF (void) {
	/* Computes the minimum based on magnitudes of input directly without LPF */
	for (idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		if(thisMag[idxFreq] < mag1[idxFreq]){
			mag1[idxFreq]=thisMag[idxFreq];
		}
	}
}

void estimateNoiseLPF(void) {
	/* Noise subtraction on LPF version of noise */
	/* Finds minimum estimate of noise from previous buffers*/
	float minMag;
	for(idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		minMag = mag1[idxFreq]; /* initialize with the a magnitude from M1*/
		if(minMag > mag2[idxFreq]){	minMag = mag2[idxFreq];}
		if(minMag > mag3[idxFreq]){	minMag = mag3[idxFreq];}
		if(minMag > mag4[idxFreq]){	minMag = mag4[idxFreq];}
		noiseLPFMag[idxFreq] = ALPHA * minMag; //this is the N(omega)
	}
	/* Low-pass filters the noise */
	
	for(idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		if (outDelay1X[idxFreq] < killThreshold){
			noiseMag[idxFreq] = (1-KSCALE)*noiseLPFMag[idxFreq]+KSCALE*noiseMag[idxFreq]; ;
		}
		else {
			noiseMag[idxFreq] = noiseLPFMag[idxFreq];
		}
	}
}

void estimateNoiseX(void) {
	/* Noise subtraction on non-LPF version */
	float minMag;
	for(idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		minMag = mag1[idxFreq]; /* initialize with the a magnitude from M1*/
		if(minMag > mag2[idxFreq]){	minMag = mag2[idxFreq];}
		if(minMag > mag3[idxFreq]){	minMag = mag3[idxFreq];}
		if(minMag > mag4[idxFreq]){	minMag = mag4[idxFreq];}
		noiseMag[idxFreq] = ALPHA * minMag;
	}
}

void overSubtract (void){
	/* Performs oversubtract for lower frequency bins*/
	for(idxFreq = 0 ; idxFreq < freqCap ; idxFreq ++){
		noiseMag[idxFreq] = noiseMag[idxFreq]*alphamax;
	}	
}

void shiftMag(void) {
	int k;
	/* Shift the noise estimate vectors and resets M1 to X(w).*/
	for(k = 0; k < FFTLEN; k++) {
		mag4[k] = mag3[k];
		mag3[k] = mag2[k];
		mag2[k] = mag1[k]; 
		mag1[k] = thisMag[k];
	}
}

void noiseSubtract(void) {
	/* Performs noise subtraction by multiplying Y(w) = X(w)G(w)*/
	for(idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		outCplxBuf[idxFreq] = rmul(gMag[idxFreq], cplxBuf[idxFreq]);
	}
}

void noiseThreshold1 (void) {
	/* Performs thresholding based on max{LAMBDA, 1-|N|/|X|}*/
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
	/* Performs thresholding based on max{LAMBDA*|N|/|X|, 1-|N|/|X|} */
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
	/* Performs thresholding based on max{LAMBDA*|P|/|X|, 1-|N|/|X|} */
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
	/* Performs thresholding based on max{LAMBDA*|N|/|P|, 1-|N|/|P|} */
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
	/* Performs thresholding based on max{LAMBDA, 1-|N|/|P|} */
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
	/* Performs thresholding based on power: max{LAMBDA, sqrt(1-(|N|/|X|)^2} */
	float minMag;
	for(idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		minMag = sqrt(1 - square(noiseMag[idxFreq])/square(thisMag[idxFreq]));
		if(minMag <= LAMBDA*thisMag[idxFreq]){
			gMag[idxFreq] = LAMBDA*thisMag[idxFreq];
		}
		else{
			gMag[idxFreq] = minMag;
		}
	}
}

void noiseThreshold7 (void) {
	/* Performs thresholding based on power: max{LAMBDA*|N|/|X|, sqrt(1-(|N|/|X|)^2)} */
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
	/* Performs thresholding based on power: max{LAMBDA*|P|/|X|, sqrt(1-(|N|/|X|)^2)} */
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
	/* Performs thresholding based on power: max{LAMBDA*|N|/|P|, sqrt(1-(|N|/|P|)^2)} */
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
	/* Performs thresholding based on power: max{LAMBDA, sqrt(1-(|N|/|P|)^2)} */
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
	/* Converts the complex time domain back to floating point */
	/* Removes musical noise content using killThreshold */
	for (idxFreq = 0; idxFreq < FFTLEN; idxFreq++ ){
		if (outCplxBuf[idxFreq].r < killThreshold) {
				outCplxBuf[idxFreq].r = 0;
				outCplxBuf[idxFreq].i = 0;
		}
	}
	/* Take IFFT */
	ifft(FFTLEN, outCplxBuf);
	/* Convert signal represented in complex number to real number by taking real part*/
	for (idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		outframe[idxFreq] = outCplxBuf[idxFreq].r;		
	}
}

void complexToFloatDelayed (void) {
	/* Task 8: Converts the complex time domain back to floating point */
	/* Note we return Y_(t-1) instead of Y(t)*/
	for (idxFreq = 0; idxFreq < FFTLEN; idxFreq++ ){
		/*|N|/|X| for time Y_(t-1)*/
		ratio  = outDelay1Ratio[idxFreq];
		/* Replace Y_(t-1) by minimum of Y_(t-2),Y_(t-1),Y_(t): outDelay2, outDelay1, outCplxBuf */
		if(ratio > musicalThreshold && outDelay1X[idxFreq] > noiseThreshold){			
			outDelay1[idxFreq] = minOfThree(outDelay2[idxFreq],outDelay1[idxFreq],outCplxBuf[idxFreq]);
		}
	}
	for (idxFreq = 0; idxFreq < FFTLEN; idxFreq++ ){
		//Store |N|/|X| for Y_(t-1)
		outDelay1Ratio[idxFreq] = noiseMag[idxFreq]/thisMag[idxFreq];
		outDelay1X[idxFreq] = thisMag[idxFreq];
	} 
	/* Removes musical noise content using killThreshold */
	for (idxFreq = 0; idxFreq < FFTLEN; idxFreq++ ){
		if (outDelay1[idxFreq].r < killThreshold) {
				outDelay1[idxFreq].r = 0;
				outDelay1[idxFreq].i = 0;
			}

	}
	
	/* Take IFFT*/
	ifft(FFTLEN, outDelay1);	
	/* Returns the processed but delayed output as real numbers */
	for (idxFreq = 0 ; idxFreq < FFTLEN ; idxFreq ++){
		outframe[idxFreq] = outDelay1[idxFreq].r;		
	}
	/* Stores the previous two frames used to minimize musical noise */
	outDelay1 = outCplxBuf;
	outDelay2 = outDelay1;
}
