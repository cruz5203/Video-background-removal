#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>

using namespace std;

vector<cv::Mat> planes(3); //�x�s3��Mat�������q�D�A�Ω�Ϲ�����
vector<cv::Mat> sums(3); //�x�s�C�ӳq�D���M
vector<cv::Mat> xysums(6); 
vector<cv::Mat> inxysums(6);
cv::Mat sum, sqsum; //�Ω�֭p�v�����M�P����M
int image_count = 0; //�O���B�z���v���V��


//�֭p��t�����
void accumulateVariance(cv::Mat& I) {//�ŧi�@�Ө禡�ǤJ�Ȭ�Mat
	if( sum.empty() ) {//�P�_sum�O�_���� �Ū��ܶi�J��l��
		sum = cv::Mat::zeros( I.size(), CV_32FC(I.channels()) );//�Ыؤ@�Ӥj�p�PI�@�˪����s�x�}
		sqsum = cv::Mat::zeros( I.size(), CV_32FC(I.channels()) );//�Ыؤ@�Ӥj�p�PI�@�˪����s�x�}
		image_count = 0;//�N�ܼ��k�s
	}
	cv::accumulate( I, sum );//�N��J���v���i��[�`
	cv::accumulateSquare( I, sqsum );//�N��J���v���i�業��[�`
	image_count++; //�B�z���v���V��+1
}


//�p���ܲ���
void computeVariance(cv::Mat& variance) {
	double one_by_N = 1.0 / image_count;//�p��B�z�v���V�ƪ��˼�
	variance = (one_by_N * sqsum) - ((one_by_N * one_by_N) * sum.mul(sum));//�p���ܲ����x�s�Jvariance
}

//�p��зǮt
void computeStdev(cv::Mat& std__) {
	double one_by_N = 1.0 / image_count;//�p��B�z�v���V�ƪ��˼�
	cv::sqrt(((one_by_N * sqsum) -((one_by_N * one_by_N) * sum.mul(sum))), std__);//�p��зǮt�x�s�Jstd__
}

//�p�⥭����
void computeAvg(cv::Mat& av) {
	double one_by_N = 1.0 / image_count;//�p��B�z�v���V�ƪ��˼�
	av = one_by_N * sum;//�p�⥭�����x�s�Jav
}


	
// ===================================================================//


//�p��@�ܲ��Ƽҫ����D�﨤�u����
void accumulateCovariance(cv::Mat& I) {
	int i, j, n;
	//��l��
	if( sum.empty() ) {
		image_count = 0;
		sum = cv::Mat::zeros(I.size(), CV_32FC(I.channels()));
		sqsum = cv::Mat::zeros(I.size(), CV_32FC(I.channels()));
		for( i=0; i<3; i++ ) {
			// the r, g, and b sums
			sums[i]= cv::Mat::zeros( I.size(), CV_32FC1 );
		}
		for( n=0; n<6; n++ ) {
			// the rr, rg, rb, gg, gb, and bb elements
			xysums[n] = cv::Mat::zeros( I.size(), CV_32FC1 );
		}
	}
	cv::accumulate( I, sum );
	cv::accumulateSquare(I, sqsum);
	cv::split( I, planes );
	for( i=0; i<3; i++ ) {
		cv::accumulate( planes[i], sums[i] );
	}
	n = 0;
	for( i=0; i<3; i++ ) {
		// "row" of Sigma
		for( j=i; j<3; j++ ) {
			// "column" of Sigma
			cv::accumulateProduct( planes[i], planes[j], xysums[n] );
			
			n++;
		}
	}
	image_count++;
}


//�p����t�����
void computeCoariance(cv::Mat& covariance
	// a six-channel array, channels are the
	// rr, rg, rb, gg, gb, and bb elements of Sigma_xy
) {
	double one_by_N = 1.0 / image_count;
	
	// reuse the xysum arrays as storage for individual entries
	//
	int n = 0;
	for( int i=0; i<3; i++ ) {
		// "row" of Sigma
		for( int j=i; j<3; j++ ) {
			// "column" of Sigma
			xysums[n] = (one_by_N * xysums[n])
			- ((one_by_N * one_by_N) * sums[i].mul(sums[j]));
			cv::invert(xysums[n], inxysums[n], cv::DECOMP_SVD);
			n++;
		}
	}
	
	// reassemble the six individual elements into a six-channel array
	//
	cv::merge( xysums, covariance );
}

////////////////////////////////////////////////////////////////////////
/////////////Utilities to run///////////////////////////////////////////

//���U���
void help(char** argv ) {
	cout << "\n"
	<< "Compute mean and std on <#frames to train on> frames of an incoming video, then run the model\n"
	<< argv[0] <<" <#frames to train on> <avi_path/filename>\n"
	<< "For example:\n"
	<< argv[0] << " 50 ../tree.avi\n"
	<< "'a' to adjust thresholds, esc, 'q' or 'Q' to quit"
	<< endl;
}
	
//////////////  Borrowed code from example_15-02  //////////////////////

// Global storage
//
// Float, 3-channel images
//
cv::Mat image; // movie frame
cv::Mat IincovF,IcovF,IavgF, IdiffF, IhiF, IlowF; //�v���������ȡB�зǮt�B���֭ȡB�C�֭�
cv::Mat tmp, mask; //scratch and our mask

// Float, 1-channel images
//
vector<cv::Mat> Igray(3); //���μv����3�q�D
vector<cv::Mat> Ilow(3);//���ΧC�֭Ȫ�3�q�D
vector<cv::Mat> Ihi(3); //���ΰ��֭Ȫ�3�q�D

// Byte, 1-channel image
//
cv::Mat Imaskt; //�B�n�Ȧs

// Thresholds
//
float high_thresh = 21.0;  //���֭�
float low_thresh = 2.0;    //�C�֭�

//�p��f���t�x�}
void computeIncov(cv::Mat& av) {
	cv::Mat covariance;
	sum /= image_count;
	calcCovarMatrix(sum.reshape(1, sum.rows * sum.cols),covariance, cv::Mat(), cv::COVAR_NORMAL || cv::COVAR_ROWS);

	invert(covariance, av, cv::DECOMP_SVD);
}

//��l�ƹϹ����Ωһݪ��Ѽ�
void AllocateImages( const cv::Mat& I ) {
	cv::Size sz = I.size();
	IincovF = cv::Mat::zeros(sz, CV_32FC3);
	IcovF = cv::Mat::zeros(sz, CV_32FC3);
	IavgF = cv::Mat::zeros(sz, CV_32FC3 ); 
	IdiffF = cv::Mat::zeros(sz, CV_32FC3 ); 
	IhiF = cv::Mat::zeros(sz, CV_32FC3 ); 
	IlowF = cv::Mat::zeros(sz, CV_32FC3 );
	tmp = cv::Mat::zeros( sz, CV_32FC3 ); 
	Imaskt = cv::Mat( sz, CV_32FC1 ); 
}

//�]�m���֭�
void setHighThreshold( float scale ) {
	IhiF = IavgF + (IdiffF * scale);//�p�Ⱚ�֭�
	cv::split( IhiF, Ihi );//�NIhiF���ά�3�q�D�x�s�bIhi
}

//�]�m�C�֭�
void setLowThreshold( float scale ) {
	IlowF = IavgF - (IdiffF * scale);//�p��C�֭�
	cv::split( IlowF, Ilow );//�NIlowF���ά�3�q�D�x�s�bIlow
}

//�ھڲέp�H���Ыؼҫ�
void createModelsfromStats() {
	IdiffF += cv::Scalar( 0.1, 0.1, 0.1 );//�W�[�p�������ȡA�קK�X�{�ҥ~�˪p
	setHighThreshold( high_thresh);//�]�m���֭�
	setLowThreshold( low_thresh);//�]�m�C�֭�
}



//�إߤ@�ӤG�ȾB�n(0,255)�A255�N��e������
void backgroundDiff(cv::Mat& I,cv::Mat& Imask) {
	
	I.convertTo( tmp, CV_32F ); //�ഫ���B�I
	cv::split( tmp, Igray ); //���ά�3�q�D
	
	// �q�D1
	cv::inRange( Igray[0], Ilow[0], Ihi[0], Imask ); //��q�D1�i��֭ȧP�_�A�b�d�򤺳]�m��255�A���b�d�򤺳]�m��0

	// �q�D2
	cv::inRange( Igray[1], Ilow[1], Ihi[1], Imaskt );//��q�D2�i��֭ȧP�_�A�b�d�򤺳]�m��255�A���b�d�򤺳]�m��0
	Imask = cv::min( Imask, Imaskt ); //�զX�q�D1�M�q�D2��mask�A�C�Ӧ�m�]�t��̤����p���ȡC

	// �q�D3
	cv::inRange( Igray[2], Ilow[2], Ihi[2], Imaskt );//��q�D3�i��֭ȧP�_�A�b�d�򤺳]�m��255�A���b�d�򤺳]�m��0
	Imask = cv::min( Imask, Imaskt );//�զX�q�D1+2�M�q�D3��mask�A�C�Ӧ�m�]�t��̤����p���ȡC

	
	Imask = 255 - Imask;// �N0�M255����A�e�����զ�I�����¦�
}

//�ϥ�mahalanobis�إߤ@�ӤG�ȾB�n
void backgroundDiff_by_mahalanobis(cv::Mat& I, cv::Mat& Imask) {
	cv::Mat mahalanobisDistances;
	for (int i = 0; i < I.rows; ++i) {
		for (int j = 0; j < I.cols; ++j) {
			cv::Vec3b pixel = I.at<cv::Vec3b>(i, j);
			cv::Point3d pixelPoint(pixel[0], pixel[1], pixel[2]);
			cv::Mat pixelMat = cv::Mat(pixel).reshape(1, 1);
			double distance = Mahalanobis(pixelMat, IavgF, IincovF);//�P�_�C�ӹ�����mahalanobis�Z��
			mahalanobisDistances.at<double>(i, j) = distance;//�N�Z���x�s�bmahalanobisDistances�x�}
		}
	}
	cv::inRange(mahalanobisDistances, high_thresh, low_thresh, Imaskt);//�P�_�O�_�b�֭Ƚd��
}

//�N�e���H�������
void showForgroundInRed( char** argv, const cv::Mat &img) {
		cv::Mat rawImage;//�ŧirawImage�Ω��x�s�B�z�᪺�v��
		cv::split( img, Igray );//�N��l�v�����ά�3�q�D
		Igray[2] = cv::max( mask, Igray[2] );//�N�B�n�P����q�D�v��������A���̤j��
		cv::merge( Igray, rawImage );//�N3�q�D�X���x�s�brawImage
		cv::imshow( argv[0], rawImage );//��ܼv���bargv[0]���f
		cv::imshow("Segmentation", mask);//��ܼv���bSegmentation���f
}

//�վ�֭�
void adjustThresholds(char** argv, cv::Mat &img) {
	int key = 1;
	while((key = cv::waitKey()) != 27 && key != 'Q' && key != 'q')  //�ˬd�O�_��J�JEscape�Bq�BQ�A�O�h�h�X�j��
	{
		if(key == 'L') { low_thresh += 0.2;}//�N�C�֭ȼW�[0.2
		if(key == 'l') { low_thresh -= 0.2;}//�N�C�֭ȴ��0.2
		if(key == 'H') { high_thresh += 0.2;}//�N���֭ȼW�[0.2
		if(key == 'h') { high_thresh -= 0.2;}//�N���֭ȴ��0.2
		cout << "H or h, L or l, esq or q to quit;  high_thresh = " << high_thresh << ", " << "low_thresh = " << low_thresh << endl;//�L�X�ثe�����֭ȻP�C�֭�
		setHighThreshold(high_thresh);//���]���֭�
		setLowThreshold(low_thresh);//���]�C�֭�
		backgroundDiff(img, mask);//�إߤ@�ӤG�ȾB�n
		showForgroundInRed(argv, img);//�N�e���H�������
	}
}

int main( int argc, char** argv) {//�D�{��
	cv::namedWindow( argv[0], cv::WINDOW_AUTOSIZE );//�Ыؤ@��window���D���ɦW�j�p�۰ʽվ�
	cv::VideoCapture cap;//�Ыؤ@�� VideoCapture ��H cap �x�s�v��
	if((argc < 3) || !cap.open(argv[2])) {//�P�_�O�_�R�O�ѼƬO�_�p��T�μv���O�_�}�ҥ��Ѧp�G�O�h����{��
		cerr << "Couldn't run the program" << endl;//��X���~����
		help(argv);//��X�{�������U�T��
		//cap.open(0);
		return -1;//�����{��
	}
	int number_to_train_on = atoi( argv[1] );//�x�s�n�V�m���v���V�ƶq

	// FIRST PROCESSING LOOP (TRAINING):
	//
	int image_count = 0;//�����v���V�ưO��
	int key;
	bool first_frame = true;
	cout << "Total frames to train on = " << number_to_train_on << endl; //��X�n�V�m���V��
	while(1) {//�إ߭I���ҫ�
		cout << "frame#: " << image_count << endl;//�L�X�p���ĴX�V
		cap >> image;//�q�v����Ū���@�V���x�s�bimage
		if( !image.data ) exit(1); //�ˬd�O�_Ū�����\���ѫh�h�X�j��
		if(image_count == 0) AllocateImages( image );//�ˬd�v���V�ưO�ƬO�_���s�A���s�h��l�ƭp��Ѽ�
		//accumulateVariance(image);//�p��Ϲ�����t
		accumulateCovariance(image);//�p��@�ܲ��Ƽҫ����D�﨤�u����
		cv::imshow( argv[0], image );//�L�X�Ϲ�
		image_count++;//�v���V�ưO��+1
		if( (key = cv::waitKey(7)) == 27 || key == 'q' || key == 'Q' || image_count >= number_to_train_on) break; //�ˬd�O�_��JEscape�Bq�BQ�έp�⧹���A�O�h�h�X�j��
	}

	// We have accumulated our training, now create the models
	//
	cout << "Creating the background model" << endl;//�L�X�I���ҫ��إߧ���
	computeAvg(IavgF);//�p��ҫ���������
	computeStdev(IdiffF);//�p��ҫ����зǮt
	computeCoariance(IcovF);//�p����t
	//computeIncov(IincovF);//�p��f���t�x�}
	createModelsfromStats();
	cout << "Done!  Hit any key to continue into single step. Hit 'a' or 'A' to adjust thresholds, esq, 'q' or 'Q' to quit\n" << endl;

	cv::namedWindow("Segmentation", cv::WINDOW_AUTOSIZE );//�Ыؤ@��window���D��Segmentation�j�p�۰ʽվ�
	while((key = cv::waitKey()) != 27 || key == 'q' || key == 'Q'  ) { //�ˬd�O�_��Jesc�A�O�h�h�X�j��
		cap >> image;//�q�v����Ū���@�V���x�s�bimage
		if( !image.data ) exit(0);//�ˬdimage�O�_Ū�����\���ѫh�h�X�j��
		cout <<  image_count++ << endl; //�N�v���V�ưO��+1�æL�X
		backgroundDiff( image, mask ); //�إߤ@�ӤG�ȾB�n
		backgroundDiff_by_mahalanobis(image, mask);//�ϥ�mahalanobis�إߤ@�ӤG�ȾB�n
		cv::imshow("Segmentation", mask); //��ܤG�ȾB�n�bSegmentation���f
		
		showForgroundInRed( argv, image);//�N�e���H�������
		if(key == 'a') {//�ˬd�O�_��Ja�A�O�h�i�J�վ�֭Ȩ禡
			cout << "In adjust thresholds, 'H' or 'h' == high thresh up or down; 'L' or 'l' for low thresh up or down." << endl;
			cout << " esq, 'q' or 'Q' to quit " << endl;
			adjustThresholds(argv, image);//�վ�֭�
			cout << "Done with adjustThreshold, back to frame stepping, esq, q or Q to quit." << endl;
		}
	}
	exit(0);
}

	
