#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>

using namespace std;

vector<cv::Mat> planes(3); //儲存3個Mat類型的通道，用於圖像分割
vector<cv::Mat> sums(3); //儲存每個通道的和
vector<cv::Mat> xysums(6); 
vector<cv::Mat> inxysums(6);
cv::Mat sum, sqsum; //用於累計影像的和與平方和
int image_count = 0; //記錄處理的影像幀數


//累計方差的函數
void accumulateVariance(cv::Mat& I) {//宣告一個函式傳入值為Mat
	if( sum.empty() ) {//判斷sum是否為空 空的話進入初始化
		sum = cv::Mat::zeros( I.size(), CV_32FC(I.channels()) );//創建一個大小與I一樣的全零矩陣
		sqsum = cv::Mat::zeros( I.size(), CV_32FC(I.channels()) );//創建一個大小與I一樣的全零矩陣
		image_count = 0;//將變數歸零
	}
	cv::accumulate( I, sum );//將輸入的影像進行加總
	cv::accumulateSquare( I, sqsum );//將輸入的影像進行平方加總
	image_count++; //處理的影像幀數+1
}


//計算變異數
void computeVariance(cv::Mat& variance) {
	double one_by_N = 1.0 / image_count;//計算處理影像幀數的倒數
	variance = (one_by_N * sqsum) - ((one_by_N * one_by_N) * sum.mul(sum));//計算變異數儲存入variance
}

//計算標準差
void computeStdev(cv::Mat& std__) {
	double one_by_N = 1.0 / image_count;//計算處理影像幀數的倒數
	cv::sqrt(((one_by_N * sqsum) -((one_by_N * one_by_N) * sum.mul(sum))), std__);//計算標準差儲存入std__
}

//計算平均值
void computeAvg(cv::Mat& av) {
	double one_by_N = 1.0 / image_count;//計算處理影像幀數的倒數
	av = one_by_N * sum;//計算平均值儲存入av
}


	
// ===================================================================//


//計算共變異數模型的非對角線元素
void accumulateCovariance(cv::Mat& I) {
	int i, j, n;
	//初始化
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


//計算協方差的函數
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

//幫助函數
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
cv::Mat IincovF,IcovF,IavgF, IdiffF, IhiF, IlowF; //影像的平均值、標準差、高閥值、低閥值
cv::Mat tmp, mask; //scratch and our mask

// Float, 1-channel images
//
vector<cv::Mat> Igray(3); //分割影像的3通道
vector<cv::Mat> Ilow(3);//分割低閥值的3通道
vector<cv::Mat> Ihi(3); //分割高閥值的3通道

// Byte, 1-channel image
//
cv::Mat Imaskt; //遮罩暫存

// Thresholds
//
float high_thresh = 21.0;  //高閥值
float low_thresh = 2.0;    //低閥值

//計算逆協方差矩陣
void computeIncov(cv::Mat& av) {
	cv::Mat covariance;
	sum /= image_count;
	calcCovarMatrix(sum.reshape(1, sum.rows * sum.cols),covariance, cv::Mat(), cv::COVAR_NORMAL || cv::COVAR_ROWS);

	invert(covariance, av, cv::DECOMP_SVD);
}

//初始化圖像分割所需的參數
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

//設置高閥值
void setHighThreshold( float scale ) {
	IhiF = IavgF + (IdiffF * scale);//計算高閥值
	cv::split( IhiF, Ihi );//將IhiF分割為3通道儲存在Ihi
}

//設置低閥值
void setLowThreshold( float scale ) {
	IlowF = IavgF - (IdiffF * scale);//計算低閥值
	cv::split( IlowF, Ilow );//將IlowF分割為3通道儲存在Ilow
}

//根據統計信息創建模型
void createModelsfromStats() {
	IdiffF += cv::Scalar( 0.1, 0.1, 0.1 );//增加小的偏移值，避免出現例外裝況
	setHighThreshold( high_thresh);//設置高閥值
	setLowThreshold( low_thresh);//設置低閥值
}



//建立一個二值遮罩(0,255)，255代表前景像素
void backgroundDiff(cv::Mat& I,cv::Mat& Imask) {
	
	I.convertTo( tmp, CV_32F ); //轉換為浮點
	cv::split( tmp, Igray ); //分割為3通道
	
	// 通道1
	cv::inRange( Igray[0], Ilow[0], Ihi[0], Imask ); //對通道1進行閥值判斷，在範圍內設置為255，不在範圍內設置為0

	// 通道2
	cv::inRange( Igray[1], Ilow[1], Ihi[1], Imaskt );//對通道2進行閥值判斷，在範圍內設置為255，不在範圍內設置為0
	Imask = cv::min( Imask, Imaskt ); //組合通道1和通道2的mask，每個位置包含兩者中較小的值。

	// 通道3
	cv::inRange( Igray[2], Ilow[2], Ihi[2], Imaskt );//對通道3進行閥值判斷，在範圍內設置為255，不在範圍內設置為0
	Imask = cv::min( Imask, Imaskt );//組合通道1+2和通道3的mask，每個位置包含兩者中較小的值。

	
	Imask = 255 - Imask;// 將0和255反轉，前景為白色背景為黑色
}

//使用mahalanobis建立一個二值遮罩
void backgroundDiff_by_mahalanobis(cv::Mat& I, cv::Mat& Imask) {
	cv::Mat mahalanobisDistances;
	for (int i = 0; i < I.rows; ++i) {
		for (int j = 0; j < I.cols; ++j) {
			cv::Vec3b pixel = I.at<cv::Vec3b>(i, j);
			cv::Point3d pixelPoint(pixel[0], pixel[1], pixel[2]);
			cv::Mat pixelMat = cv::Mat(pixel).reshape(1, 1);
			double distance = Mahalanobis(pixelMat, IavgF, IincovF);//判斷每個像素的mahalanobis距離
			mahalanobisDistances.at<double>(i, j) = distance;//將距離儲存在mahalanobisDistances矩陣
		}
	}
	cv::inRange(mahalanobisDistances, high_thresh, low_thresh, Imaskt);//判斷是否在閥值範圍內
}

//將前景以紅色顯示
void showForgroundInRed( char** argv, const cv::Mat &img) {
		cv::Mat rawImage;//宣告rawImage用於儲存處理後的影像
		cv::split( img, Igray );//將原始影像分割為3通道
		Igray[2] = cv::max( mask, Igray[2] );//將遮罩與紅色通道逐元素比較，取最大值
		cv::merge( Igray, rawImage );//將3通道合併儲存在rawImage
		cv::imshow( argv[0], rawImage );//顯示影像在argv[0]窗口
		cv::imshow("Segmentation", mask);//顯示影像在Segmentation窗口
}

//調整閥值
void adjustThresholds(char** argv, cv::Mat &img) {
	int key = 1;
	while((key = cv::waitKey()) != 27 && key != 'Q' && key != 'q')  //檢查是否輸入入Escape、q、Q，是則退出迴圈
	{
		if(key == 'L') { low_thresh += 0.2;}//將低閥值增加0.2
		if(key == 'l') { low_thresh -= 0.2;}//將低閥值減少0.2
		if(key == 'H') { high_thresh += 0.2;}//將高閥值增加0.2
		if(key == 'h') { high_thresh -= 0.2;}//將高閥值減少0.2
		cout << "H or h, L or l, esq or q to quit;  high_thresh = " << high_thresh << ", " << "low_thresh = " << low_thresh << endl;//印出目前的高閥值與低閥值
		setHighThreshold(high_thresh);//重設高閥值
		setLowThreshold(low_thresh);//重設低閥值
		backgroundDiff(img, mask);//建立一個二值遮罩
		showForgroundInRed(argv, img);//將前景以紅色顯示
	}
}

int main( int argc, char** argv) {//主程式
	cv::namedWindow( argv[0], cv::WINDOW_AUTOSIZE );//創建一個window標題為檔名大小自動調整
	cv::VideoCapture cap;//創建一個 VideoCapture 對象 cap 儲存影片
	if((argc < 3) || !cap.open(argv[2])) {//判斷是否命令參數是否小於三或影片是否開啟失敗如果是則中止程式
		cerr << "Couldn't run the program" << endl;//輸出錯誤消息
		help(argv);//輸出程式的幫助訊息
		//cap.open(0);
		return -1;//結束程式
	}
	int number_to_train_on = atoi( argv[1] );//儲存要訓練的影片幀數量

	// FIRST PROCESSING LOOP (TRAINING):
	//
	int image_count = 0;//紀錄影像幀數記數
	int key;
	bool first_frame = true;
	cout << "Total frames to train on = " << number_to_train_on << endl; //輸出要訓練的幀數
	while(1) {//建立背景模型
		cout << "frame#: " << image_count << endl;//印出計算到第幾幀
		cap >> image;//從影像中讀取一幀並儲存在image
		if( !image.data ) exit(1); //檢查是否讀取成功失敗則退出迴圈
		if(image_count == 0) AllocateImages( image );//檢查影像幀數記數是否為零，為零則初始化計算參數
		//accumulateVariance(image);//計算圖像的方差
		accumulateCovariance(image);//計算共變異數模型的非對角線元素
		cv::imshow( argv[0], image );//印出圖像
		image_count++;//影像幀數記數+1
		if( (key = cv::waitKey(7)) == 27 || key == 'q' || key == 'Q' || image_count >= number_to_train_on) break; //檢查是否輸入Escape、q、Q或計算完成，是則退出迴圈
	}

	// We have accumulated our training, now create the models
	//
	cout << "Creating the background model" << endl;//印出背景模型建立完畢
	computeAvg(IavgF);//計算模型的平均值
	computeStdev(IdiffF);//計算模型的標準差
	computeCoariance(IcovF);//計算協方差
	//computeIncov(IincovF);//計算逆協方差矩陣
	createModelsfromStats();
	cout << "Done!  Hit any key to continue into single step. Hit 'a' or 'A' to adjust thresholds, esq, 'q' or 'Q' to quit\n" << endl;

	cv::namedWindow("Segmentation", cv::WINDOW_AUTOSIZE );//創建一個window標題為Segmentation大小自動調整
	while((key = cv::waitKey()) != 27 || key == 'q' || key == 'Q'  ) { //檢查是否輸入esc，是則退出迴圈
		cap >> image;//從影像中讀取一幀並儲存在image
		if( !image.data ) exit(0);//檢查image是否讀取成功失敗則退出迴圈
		cout <<  image_count++ << endl; //將影像幀數記數+1並印出
		backgroundDiff( image, mask ); //建立一個二值遮罩
		backgroundDiff_by_mahalanobis(image, mask);//使用mahalanobis建立一個二值遮罩
		cv::imshow("Segmentation", mask); //顯示二值遮罩在Segmentation窗口
		
		showForgroundInRed( argv, image);//將前景以紅色顯示
		if(key == 'a') {//檢查是否輸入a，是則進入調整閥值函式
			cout << "In adjust thresholds, 'H' or 'h' == high thresh up or down; 'L' or 'l' for low thresh up or down." << endl;
			cout << " esq, 'q' or 'Q' to quit " << endl;
			adjustThresholds(argv, image);//調整閥值
			cout << "Done with adjustThreshold, back to frame stepping, esq, q or Q to quit." << endl;
		}
	}
	exit(0);
}

	
