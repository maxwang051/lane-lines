package sd.streaming;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.ImageView;

import org.apache.commons.math3.fitting.PolynomialCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoints;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.lang.Math;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity{
    private static final String TAG = "OCVSample::Activity";

    int count = 0;
    ImageView imageView;

    Mat img, warpSrc, warpDst, M, Minv, warped, gray, hls, lab, hls_l, lab_b;
    Mat l_thresh_high, b_thresh_high, sobel_x, sobel_x_low, sobel_x_high, binary_warped;
    Mat vand,ones,squared;
    Mat leftploty, rightploty, ploty, left_fitx, left_fitx2, left_fitx1, right_fitx, right_fitx2, right_fitx1, right_weights, left_weights;
    Mat rightx, righty, leftx, lefty, pts_left, pts_right;
    Mat histogram, nonzero, nonzerox, nonzeroy, good_left_inds, good_right_inds;
    Mat warp_zero, color_warp, newwarp, result;

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private static final String MODEL_FILE = "file:///android_asset/model1.pb";
    private static final String INPUT_NODE = "lambda_input_1";
    private static final String OUTPUT_NODE = "add_8";
    private TensorFlowInferenceInterface inferenceInterface;
    private static final int[] INPUT_SIZE = {1,160,320,3};

    int width = 320;
    int height = 160;
    int channels = 3;

    int type = CvType.CV_32F;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    img = new Mat(height,width,type);

                    warpSrc = new Mat();
                    warpSrc.create(4, 2, type);

                    warpDst = new Mat();
                    warpDst.create(4, 2, type);

                    M = new Mat();
                    M.create(3,3, type);
                    Minv = new Mat();
                    warped = new Mat(0, 0, type);

                    gray = new Mat(0, 0, type);
                    hls = new Mat(0, 0, type);
                    lab = new Mat(0, 0, type);

                    hls_l = new Mat(0, 0, type);
                    lab_b = new Mat(0, 0, type);

                    l_thresh_high = new Mat(0, 0, type);
                    b_thresh_high = new Mat(0, 0, type);
                    sobel_x = new Mat(0, 0, type);
                    sobel_x_low = new Mat(0, 0, type);
                    sobel_x_high = new Mat(0, 0, type);
                    binary_warped = new Mat(0, 0, type);

                    histogram = new Mat(0, 0, type);
                    nonzero = new Mat(0, 0, type);
                    nonzerox = new Mat(0, 0, type);
                    nonzeroy = new Mat(0, 0, type);
                    good_left_inds = new Mat(0, 0, type);
                    good_right_inds = new Mat(0, 0, type);
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    Timer timer;
    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

        count = 0;

        timer = new Timer();
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                nextImage();
            }

        }, 0, 100);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        count = 0;
        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);
    }

    String packageName;
    String drawable = "drawable";
    int id;
    Bitmap bm;

    void nextImage() {
        packageName = getPackageName();
        id = getResources().getIdentifier("c" + Integer.toString(count),
                drawable, packageName);
        imageView = (ImageView) findViewById(R.id.imageView);

        try {
            img = Utils.loadResource(MainActivity.this, id);
        } catch (IOException e) {
            Log.e(TAG,Log.getStackTraceString(e));
        }

        // Create matrices for these for some reason
        //warped.create(img.rows(), img.cols(), CvType.CV_32F);
        //hls_l.create(img.rows(), img.cols(), CvType.CV_32F);
        //lab_b.create(img.rows(), img.cols(), CvType.CV_32F);
        //l_thresh_high.create(img.rows(), img.cols(), CvType.CV_32F);

        processImage(img);

        bm = Bitmap.createBitmap(width, height,Bitmap.Config.ARGB_8888);

        Utils.matToBitmap(result, bm);

        runOnUiThread(new Runnable() {
            public void run() {
                imageView.setImageBitmap(bm);
            }
        });


        count = (count + 1) % 100;
    }

    void polyfit(Mat X, Mat Y, Mat weights, int type){
        vand = new Mat(1,X.cols(),type);
        Core.multiply(X, X, vand );
        vand.push_back(X);
        ones = new Mat(1,X.cols(),type, new Scalar(1));
        vand.push_back(ones);
        squared = new Mat(vand.rows(),vand.rows(),type);
        Core.gemm(vand, vand.t(),1,new Mat(0,0,5),0,squared);
        Core.gemm(squared.inv(),vand,1,new Mat(0,0,5),0,vand);
        Core.gemm(vand,Y,1,new Mat(0,0,5),0,weights);
    }

    float[] inputFloats;

    int offset = 40;

    int midpoint = 160;
    int leftx_base = 0;
    int rightx_base = 0;
    int num1,num2;

    // Number of sliding windows
    int nwindows = 10;
    // Set height of windows
    int window_height = 16;
    // Set the width of the windows +/- margin
    int margin = 40;
    // Set minimum number of pixels found to recenter window
    int minpix = 10;

    int win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high;

    int leftx_current, rightx_current;
    int num_nonzeroes;

    int right_count, left_count;    //count of number of nonzeros in the window
    double right_sum, left_sum;   //sum of the indeces of nonzeros

    double x,y;

    int leftWidth, rightWidth;

    List<Mat> layers = new ArrayList<>();
    List<Point> pts_left_list = new ArrayList<>();
    List<Point> pts_right_list = new ArrayList<>();
    List<Mat> matrices = new ArrayList<>();
    List<MatOfPoint> listOfPoints = new ArrayList<>();

    MatOfPoint matOfPointLeft, matOfPointRight, matOfPointPath;
    MatOfFloat rgb;

    double[] green = {0, 255, 0};
    double[] red = {255,0,0};
    float[] resu = {0};
    double angle;
    List<Point> path = new ArrayList<>();
    int pathMaxHeight = 60;

    void processImage(Mat img) {
        // Warp perspective to top down
        warpSrc.put(0, 0, new double[]{110,40});
        warpSrc.put(1, 0, new double[]{210,40});
        warpSrc.put(2, 0, new double[]{20,130});
        warpSrc.put(3, 0, new double[]{315,130});

        warpDst.put(0, 0, new double[]{offset,0});
        warpDst.put(1, 0, new double[]{width-offset,0});
        warpDst.put(2, 0, new double[]{offset,height});
        warpDst.put(3, 0, new double[]{width-offset,height});

        M = Imgproc.getPerspectiveTransform(warpSrc, warpDst);
        Minv = Imgproc.getPerspectiveTransform(warpDst, warpSrc);

        Imgproc.warpPerspective(img, warped, M, new Size(width, height), Imgproc.INTER_LINEAR);

        // Perform color transformations
        Imgproc.cvtColor(warped, gray, Imgproc.COLOR_RGB2GRAY);

        // Get Sobel in x direction
        Imgproc.Sobel(gray, sobel_x, -1, 1, 0);
        // Absolute value of Sobel
        Core.absdiff(sobel_x, new Scalar(0), sobel_x);
        // Normalize Sobel
        Core.MinMaxLocResult sobel_minMax = Core.minMaxLoc(sobel_x);
        Core.multiply(sobel_x, new Scalar(255.0 / sobel_minMax.maxVal), sobel_x);
        // Threshold sobel values greater than 50 and less than 100 and combine back into sobel_x
        Imgproc.threshold(sobel_x, sobel_x_low, 70, 255, Imgproc.THRESH_BINARY);
        Imgproc.threshold(sobel_x, sobel_x_high, 255, 255, Imgproc.THRESH_BINARY_INV);
        Core.bitwise_and(sobel_x_low, sobel_x_high, sobel_x);

        //Core.bitwise_or(l_thresh_high, sobel_x, binary_warped);
        //Core.bitwise_or(binary_warped, b_thresh_high, binary_warped);

        //Imgproc.threshold(binary_warped, binary_warped, 150, 255, Imgproc.THRESH_BINARY);

        binary_warped = sobel_x.clone();

        // POLYFIT

        // Single row vector where each element is the sum of the corresponding column
        // Sum up bottom half of image
        Mat temp = binary_warped.rowRange(binary_warped.height() / 2, binary_warped.height());
        Core.reduce(temp, histogram, 0, Core.REDUCE_SUM, histogram.depth());

        // Get peaks for left and right halves of image
        leftx_base = 0;
        rightx_base = 0;

        for (int i = 0; i < midpoint; i++) {
            num1 = (int) histogram.get(0, i)[0];
            num2 = (int) histogram.get(0, i + midpoint)[0];

            if (num1 > leftx_base) leftx_base = i;
            if (num2 > rightx_base) rightx_base = i+midpoint;
        }

        // Get nonzero pixels and separate into x and y coordinates
        Core.findNonZero(binary_warped, nonzero);
        Core.extractChannel(nonzero, nonzerox, 0);
        Core.extractChannel(nonzero, nonzeroy, 1);

        // Current positions to be updated for each window
        leftx_current = leftx_base;
        rightx_current = rightx_base;

        num_nonzeroes = nonzero.rows();

        rightx = new Mat(0,1,type);
        righty = new Mat(0,1,type);
        leftx = new Mat(0,1,type);
        lefty = new Mat(0,1,type);

        // Step through the windows one by one
        for (int i = 0; i < nwindows; i++) {
            win_y_low = binary_warped.height() - (i+1) * window_height;
            win_y_high = binary_warped.height() - i * window_height;
            win_xleft_low = leftx_current - margin;
            win_xleft_high = leftx_current + margin;
            win_xright_low = rightx_current - margin;
            win_xright_high = rightx_current + margin;

            right_count = 0;    //count of number of nonzeros in the window
            right_sum = 0;   //sum of the indeces of nonzeros
            left_count = 0;
            left_sum = 0;

            //Identify the nonzero pixels in x and y within the window
            for(int j = 0; j < num_nonzeroes; j++){     //iterate through each nonzero
                x = nonzerox.get(j,0)[0];
                y = nonzeroy.get(j,0)[0];
                if((y >= win_y_low) && (y < win_y_high) && (x >= win_xleft_low) &&  (x < win_xleft_high)) {

                    left_count++;
                    left_sum+=x;
                    //Append these indices to the lists
                    //Extract left and right line pixel positions
                    leftx.push_back(new Mat(1,1,type,new Scalar(x)));
                    lefty.push_back(new Mat(1,1,type,new Scalar(y)));
                    }
                if((y >= win_y_low) && (y < win_y_high) && (x >= win_xright_low) &&  (x < win_xright_high)) {

                    right_count++;
                    right_sum+=x;
                    //Append these indices to the lists
                    //Extract left and right line pixel positions
                    rightx.push_back(new Mat(1,1,type,new Scalar(x)));
                    righty.push_back(new Mat(1,1,type,new Scalar(y)));

                }
            }
            //If you found > minpix pixels, recenter next window on their mean position
            if(right_count > minpix){
                rightx_current= (int)(right_sum / right_count);
            }
            if(left_count > minpix){
                leftx_current= (int)(left_sum / left_count);
            }

            //Imgproc.threshold(nonzerox)
        }

        right_weights = new Mat(3,1,type);
        left_weights = new Mat(3,1,type);

        //Fit a second order polynomial to each
        PolynomialCurveFitter fitter = PolynomialCurveFitter.create(2);
        if(rightx.rows() > 0){
            polyfit(righty.t(),rightx,right_weights,5);
        }
        if(leftx.rows() > 0){
            polyfit(lefty.t(),leftx,left_weights,5);
        }

        // DRAW ON LANE

        warp_zero = new Mat(160, 320, CvType.CV_8U, new Scalar(0));
        layers.clear();
        layers.add(warp_zero);
        layers.add(warp_zero);
        layers.add(warp_zero);
        color_warp = new Mat(0, 0, CvType.CV_32F);
        Core.merge(layers, color_warp);

        //Mat ploty = new Mat(1, width, CvType.CV_32F);
        //for (int i = 0; i < width; i++) { ploty.put(0, i, i);}

        Core.MinMaxLocResult leftmm = Core.minMaxLoc(leftx);
        //leftWidth = (int)(leftmm.maxVal);
        leftWidth = height;
        rightWidth = height;
        leftploty = new Mat(1,leftWidth,CvType.CV_32F);
        for (int i = 0; i < leftWidth; i++ ){leftploty.put(0,i,i);}

        Core.MinMaxLocResult rightmm = Core.minMaxLoc(rightx);
        //rightWidth = (int)(width - rightmm.minVal);
        //leftWidth = width;
        //rightWidth = width;
        rightploty = new Mat(1,rightWidth,CvType.CV_32F);
        for (int i = 0; i < rightWidth; i++ ){rightploty.put(0,i,i);}

        left_fitx = new Mat(0, 0, CvType.CV_32F);
        left_fitx2 = new Mat(0, 0, CvType.CV_32F);
        left_fitx1 = new Mat(0, 0, CvType.CV_32F);

        right_fitx = new Mat(0, 0, CvType.CV_32F);
        right_fitx2 = new Mat(0, 0, CvType.CV_32F);
        right_fitx1 = new Mat(0, 0, CvType.CV_32F);

        // left_fit[0]*ploty**2
        Core.pow(leftploty, 2, left_fitx2);
        Core.multiply(left_fitx2, new Scalar(left_weights.get(0, 0)[0]), left_fitx2);
        // left_fit[1]*ploty
        Core.multiply(leftploty, new Scalar(left_weights.get(1, 0)[0]), left_fitx1);
        Core.add(left_fitx2, left_fitx1, left_fitx);
        Core.add(left_fitx, new Scalar(left_weights.get(2, 0)[0]), left_fitx);

        // right_fit[0]*ploty**2
        Core.pow(rightploty, 2, right_fitx2);
        Core.multiply(right_fitx2, new Scalar(right_weights.get(0, 0)[0]), right_fitx2);
        // right_fit[1]*ploty
        Core.multiply(rightploty, new Scalar(right_weights.get(1, 0)[0]), right_fitx1);
        Core.add(right_fitx2, right_fitx1, right_fitx);
        Core.add(right_fitx, new Scalar(right_weights.get(2, 0)[0]), right_fitx);

        // Get points for left line
        matrices.clear();
        matrices.add(left_fitx);
        matrices.add(leftploty);
        pts_left = new Mat(0, 0, CvType.CV_32F);
        Core.vconcat(matrices, pts_left);
        Core.transpose(pts_left, pts_left);
        Core.flip(pts_left, pts_left, -1);

        // Get points for right line
        matrices.clear();
        matrices.add(right_fitx);
        matrices.add(rightploty);
        pts_right = new Mat(0, 0, CvType.CV_32F);
        Core.vconcat(matrices, pts_right);
        Core.transpose(pts_right, pts_right);
        Core.flip(pts_right, pts_right, -1);

        // Draw lines
        pts_left_list.clear();
        pts_right_list.clear();
        for (int i = 0; i < leftWidth; i++) {
            Point p = new Point(pts_left.get(i, 1)[0], pts_left.get(i, 0)[0]);
            pts_left_list.add(p);
            //Log.d("point", p.x + " " + p.y);
        }
        for (int i = 0; i < rightWidth; i++) {
            Point p = new Point(pts_right.get(i, 1)[0], pts_right.get(i, 0)[0]);
            pts_right_list.add(p);
            //Log.d("point", p.x + " " + p.y);
        }

        matOfPointLeft = new MatOfPoint();
        matOfPointRight = new MatOfPoint();
        matOfPointLeft.fromList(pts_left_list);
        matOfPointRight.fromList(pts_right_list);

        listOfPoints.clear();
        listOfPoints.add(matOfPointLeft);
        Imgproc.polylines(color_warp, listOfPoints, false, new Scalar(green), 5);

        listOfPoints.clear();
        listOfPoints.add(matOfPointRight);
        Imgproc.polylines(color_warp, listOfPoints, false, new Scalar(green), 5);

        newwarp = new Mat();
        Imgproc.warpPerspective(color_warp, newwarp, Minv, new Size(width, height), Imgproc.INTER_LINEAR);

        rgb = new MatOfFloat(CvType.CV_32F);
        img.convertTo(rgb,CvType.CV_32F);
        inputFloats = new float[(int)(channels*width*height)];
        rgb.get(0,0,inputFloats);

        inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SIZE, inputFloats);
        inferenceInterface.runInference(new String[] {OUTPUT_NODE});
        inferenceInterface.readNodeFloat(OUTPUT_NODE, resu);
        //Log.d("angle",Float.toString(resu[0]));
        //angle = (double)resu[0] + 360;

        //angle = 80; //get from NN
        //double slip_fator = 0.0014; // slip factor obtained from real data
        //double steer_ratio = 15.3;  // from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
        //double wheel_base = 2.67;   //from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

        // Convert angle from degrees to rads
        angle = Math.PI * resu[0] / 180;

        // Draw line from bottom middle to end point of line
        Point start = new Point(width/2.0, height);
        Point end = new Point(width/2.0 + 60 * Math.sin(angle), height - 60 * Math.cos(angle));
        Imgproc.line(img, start, end, new Scalar(0, 0, 255));

        result = new Mat();

        Core.addWeighted(img, 1, newwarp, 0.5, 0, result);
    }
}
