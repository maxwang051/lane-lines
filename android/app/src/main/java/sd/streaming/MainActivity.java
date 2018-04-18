package sd.streaming;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Base64;
import android.util.Log;
import android.widget.ImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "OCVSample::Activity";

    int count = 0;
    ImageView imageView;

    List<Mat> hls_list;
    Mat img, warpSrc, warpDst, M, Minv, warped, gray, hls, lab, hls_l, lab_b;
    Mat l_thresh_high, b_thresh_high, sobel_x, sobel_x_low, sobel_x_high, binary_warped;
    Mat histogram, nonzero, nonzerox, nonzeroy, good_left_inds, good_right_inds;
    Mat result;
    LinkedList<Bitmap> imgData = new LinkedList<>();

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    img = new Mat(160,320,CvType.CV_32F);

                    warpSrc = new Mat();
                    warpSrc.create(4, 2, CvType.CV_32F);

                    warpDst = new Mat();
                    warpDst.create(4, 2, CvType.CV_32F);

                    M = new Mat();
                    M.create(3,3, CvType.CV_32F);
                    Minv = new Mat();
                    warped = new Mat(0, 0, CvType.CV_32F);

                    gray = new Mat(0, 0, CvType.CV_32F);
                    hls = new Mat(0, 0, CvType.CV_32F);
                    lab = new Mat(0, 0, CvType.CV_32F);

                    hls_l = new Mat(0, 0, CvType.CV_32F);
                    lab_b = new Mat(0, 0, CvType.CV_32F);

                    l_thresh_high = new Mat(0, 0, CvType.CV_32F);
                    b_thresh_high = new Mat(0, 0, CvType.CV_32F);
                    sobel_x = new Mat(0, 0, CvType.CV_32F);
                    sobel_x_low = new Mat(0, 0, CvType.CV_32F);
                    sobel_x_high = new Mat(0, 0, CvType.CV_32F);
                    binary_warped = new Mat(0, 0, CvType.CV_32F);

                    histogram = new Mat(0, 0, CvType.CV_32F);
                    nonzero = new Mat(0, 0, CvType.CV_32F);
                    nonzerox = new Mat(0, 0, CvType.CV_32F);
                    nonzeroy = new Mat(0, 0, CvType.CV_32F);
                    good_left_inds = new Mat(0, 0, CvType.CV_32F);
                    good_right_inds = new Mat(0, 0, CvType.CV_32F);
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

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
        try {
            String hostAddress = "10.147.33.212";
            InetAddress addr = null;
            int tcpPort = 4444;// hardcoded -- must match the server's tcp port
            BufferedReader input;
            Socket tcp;

            addr = InetAddress.getByName(hostAddress);
            tcp = new Socket(addr, tcpPort);
            input = new BufferedReader(new InputStreamReader(tcp.getInputStream()));
            String in = null;
            while((in = input.readLine()) != null){
                byte[] imagebyte;
                imagebyte = Base64.decode(in,Base64.DEFAULT);
                final Bitmap image = BitmapFactory.decodeByteArray(imagebyte, 0, imagebyte.length);
                imgData.addLast(image);
            }
        } catch (UnknownHostException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        Timer timer = new Timer();
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        nextImage();
                    }
                });
            }
        }, 0, 50);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        count = 0;
    }

    void nextImage() {
        String packageName = getPackageName();
        String type = "drawable";
        int id = getResources().getIdentifier("image" + Integer.toString(count),
                type, packageName);
        imageView = (ImageView) findViewById(R.id.imageView);

        Bitmap bmp = imgData.poll();
        if(bmp == null) {
            return;
        }
//        Bitmap bmp = BitmapFactory.decodeResource(getResources(),R.drawable.image0);

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

//        processImage(img);

//        Bitmap bm = Bitmap.createBitmap(result.cols(), result.rows(),Bitmap.Config.ARGB_8888);
//        Utils.matToBitmap(result, bm);
        imageView.setImageBitmap(bmp);

        count = (count + 1) % 64;
    }

    void polyfit(Mat X, Mat Y, Mat weights, int type){
        Mat vand = new Mat(1,X.cols(),type);
        Core.multiply(X, X, vand );
        vand.push_back(X);
        Mat ones = new Mat(1,X.cols(),type,new Scalar(1));
        vand.push_back(ones);
        Mat squared = new Mat(vand.rows(),vand.rows(),type);
        Core.gemm(vand, vand.t(),1,new Mat(0,0,5),0,squared);
        Core.gemm(squared.inv(),vand,1,new Mat(0,0,5),0,vand);
        Core.gemm(vand,Y,1,new Mat(0,0,5),0,weights);
    }

    void processImage(Mat img) {
        int height = img.height();
        int width = img.width();

        // Warp perspective to top down
        int offset = 40;

        warpSrc.put(0, 0, new double[]{110,40});
        warpSrc.put(1, 0, new double[]{210,40});
        warpSrc.put(2, 0, new double[]{20,120});
        warpSrc.put(3, 0, new double[]{300,120});

        warpDst.put(0, 0, new double[]{offset,0});
        warpDst.put(1, 0, new double[]{width-offset,0});
        warpDst.put(2, 0, new double[]{offset,height});
        warpDst.put(3, 0, new double[]{width-offset,height});

        M = Imgproc.getPerspectiveTransform(warpSrc, warpDst);
        Minv = Imgproc.getPerspectiveTransform(warpDst, warpSrc);

        Imgproc.warpPerspective(img, warped, M, new Size(width, height), Imgproc.INTER_LINEAR);

        // Perform color transformations
        Imgproc.cvtColor(warped, gray, Imgproc.COLOR_RGB2GRAY);
        Imgproc.cvtColor(warped, hls, Imgproc.COLOR_RGB2HLS);
        Imgproc.cvtColor(warped, lab, Imgproc.COLOR_RGB2Lab);

        // Get L and B channels from HLS and LAB
        Core.extractChannel(hls, hls_l, 1);
        Core.extractChannel(lab, lab_b, 2);

        // Normalize lab_b if there is yellow in the image
        Core.MinMaxLocResult b_minMax = Core.minMaxLoc(b_thresh_high);
        if (b_minMax.maxVal > 175) {
            Core.multiply(lab_b, new Scalar(255.0 / b_minMax.maxVal), lab_b);
        }

        // Threshold L and B channels
        Imgproc.threshold(hls_l, l_thresh_high, 180, 255, Imgproc.THRESH_BINARY);
        Imgproc.threshold(lab_b, b_thresh_high, 180, 255, Imgproc.THRESH_BINARY);

        // Get Sobel in x direction
        Imgproc.Sobel(gray, sobel_x, -1, 1, 0);
        // Absolute value of Sobel
        Core.absdiff(sobel_x, new Scalar(0), sobel_x);
        // Normalize Sobel
        Core.MinMaxLocResult sobel_minMax = Core.minMaxLoc(sobel_x);
        Core.multiply(sobel_x, new Scalar(255.0 / sobel_minMax.maxVal), sobel_x);
        // Threshold sobel values greater than 50 and less than 100 and combine back into sobel_x
        Imgproc.threshold(sobel_x, sobel_x_low, 50, 255, Imgproc.THRESH_BINARY);
        Imgproc.threshold(sobel_x, sobel_x_high, 140, 255, Imgproc.THRESH_BINARY_INV);
        Core.bitwise_and(sobel_x_low, sobel_x_high, sobel_x);

        Core.bitwise_or(l_thresh_high, sobel_x, binary_warped);
        Core.bitwise_or(binary_warped, b_thresh_high, binary_warped);

        Imgproc.threshold(binary_warped, binary_warped, 150, 255, Imgproc.THRESH_BINARY);


        // POLYFIT

        // Single row vector where each element is the sum of the corresponding column
        Core.reduce(binary_warped, histogram, 0, Core.REDUCE_SUM, histogram.depth());

        // Get peaks for left and right halves of image
        int midpoint = 160;
        int leftx_base = 0;
        int rightx_base = 0;

        for (int i = 0; i < midpoint; i++) {
            int num1 = (int) histogram.get(0, i)[0];
            int num2 = (int) histogram.get(0, i + midpoint)[0];

            if (num1 > leftx_base) leftx_base = i;
            if (num2 > rightx_base) rightx_base = i+midpoint;
        }

        // Number of sliding windows
        int nwindows = 8;
        // Set height of windows
        int window_height = 20;

        // Get nonzero pixels and separate into x and y coordinates
        Core.findNonZero(binary_warped, nonzero);
        Core.extractChannel(nonzero, nonzerox, 0);
        Core.extractChannel(nonzero, nonzeroy, 1);

        // Current positions to be updated for each window
        int leftx_current = leftx_base;
        int rightx_current = rightx_base;

        // Set the width of the windows +/- margin
        int margin = 40;
        // Set minimum number of pixels found to recenter window
        int minpix = 20;

        int num_nonzeroes = nonzero.rows();

        //Alec's attempt
        int type = 5;
        Mat left_lane_inds = new Mat(0,1,5); //doesn't seem necessary
        Mat right_lane_inds = new Mat(0,1,5); //doesn't seem necessary
        Mat rightx = new Mat(0,1,5);
        Mat righty = new Mat(0,1,5);
        Mat leftx = new Mat(0,1,5);
        Mat lefty = new Mat(0,1,5);
        // Step through the windows one by one
        for (int i = 0; i < nwindows; i++) {
            int win_y_low = binary_warped.height() - (i+1) * window_height;
            int win_y_high = binary_warped.height() - i * window_height;
            int win_xleft_low = leftx_current - margin;
            int win_xleft_high = leftx_current + margin;
            int win_xright_low = rightx_current - margin;
            int win_xright_high = rightx_current + margin;

            int right_count = 0;    //count of number of nonzeros in the window
            double right_sum = 0;   //sum of the indeces of nonzeros
            int left_count = 0;
            double left_sum = 0;

            //Identify the nonzero pixels in x and y within the window
            for(int j = 0; j < num_nonzeroes; j++){     //iterate through each nonzero
                double x = nonzerox.get(j,0)[0];
                double y = nonzeroy.get(j,0)[0];
                if((y >= win_y_low) && (y < win_y_high) && (x >= win_xleft_low) &&  (x < win_xleft_high)) {

                    left_count++;
                    left_sum+=x;
                    //Append these indices to the lists
                    left_lane_inds.push_back(new Mat(1,1,type,new Scalar(j)));      //doesn't seem necessary
                    //Extract left and right line pixel positions
                    leftx.push_back(new Mat(1,1,type,new Scalar(x)));
                    lefty.push_back(new Mat(1,1,type,new Scalar(y)));
                }
                if((y >= win_y_low) && (y < win_y_high) && (x >= win_xright_low) &&  (x < win_xright_high)) {

                    right_count++;
                    right_sum+=x;
                    //Append these indices to the lists
                    right_lane_inds.push_back(new Mat(1,1,type,new Scalar(j)));      //doesn't seem necessary
                    //Extract left and right line pixel positions
                    rightx.push_back(new Mat(1,1,type,new Scalar(x)));
                    righty.push_back(new Mat(1,1,type,new Scalar(y)));
                }
            }
            //If you found > minpix pixels, recenter next window on their mean position
            if(right_count > minpix){
                rightx_current= (int)(right_sum / right_count);
                Log.e("start",Integer.toString(rightx_current));
            }
            if(left_count > minpix){
                leftx_current= (int)(left_sum / left_count);
                Log.e("start",Integer.toString(leftx_current));
            }

            //Imgproc.threshold(nonzerox)
        }

        //Fit a second order polynomial to each
        Mat right_weights = new Mat(3,1,nonzero.type());
        Mat left_weights = new Mat(3,1,nonzero.type());

        if(rightx.rows() > 0){
            polyfit(rightx.t(),righty,right_weights,5);
        }
        if(leftx.rows() > 0){
            polyfit(leftx.t(),lefty,left_weights,5);
        }

        // DRAW ON LANE

        Mat warp_zero = new Mat(160, 320, CvType.CV_8U, new Scalar(0));
        List<Mat> layers = new ArrayList<>();
        layers.add(warp_zero);
        layers.add(warp_zero);
        layers.add(warp_zero);
        Mat color_warp = new Mat(0, 0, CvType.CV_32F);
        Core.merge(layers, color_warp);

        Mat leftploty = new Mat(1, width, CvType.CV_32F);
        Mat rightploty = new Mat(1, width, CvType.CV_32F);
        for (int i = 0; i < width/2; i++) { leftploty.put(0, i, i);
        rightploty.put(0,i,i + width/2);}

        Mat left_fitx = new Mat(0, 0, CvType.CV_32F);
        Mat left_fitx2 = new Mat(0, 0, CvType.CV_32F);
        Mat left_fitx1 = new Mat(0, 0, CvType.CV_32F);

        Mat right_fitx = new Mat(0, 0, CvType.CV_32F);
        Mat right_fitx2 = new Mat(0, 0, CvType.CV_32F);
        Mat right_fitx1 = new Mat(0, 0, CvType.CV_32F);

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
        List<Mat> matrices = new ArrayList<>();
        matrices.add(left_fitx);
        matrices.add(leftploty);
        Mat pts_left = new Mat(0, 0, CvType.CV_32F);
        Core.vconcat(matrices, pts_left);
        Core.transpose(pts_left, pts_left);
        Core.flip(pts_left, pts_left, 1);

        // Get points for right line
        matrices.clear();
        matrices.add(right_fitx);
        matrices.add(rightploty);
        Mat pts_right = new Mat(0, 0, CvType.CV_32F);
        Core.vconcat(matrices, pts_right);
        Core.transpose(pts_right, pts_right);
        Core.flip(pts_right, pts_right, 1);

        // Get points for both lines
        matrices.clear();
        matrices.add(pts_left);
        matrices.add(pts_right);
        Mat pts = new Mat(0, 0, CvType.CV_32F);
        Core.hconcat(matrices, pts);

        // Draw lines
        List<Point> pts_left_list = new ArrayList<>();
        List<Point> pts_right_list = new ArrayList<>();
        for (int i = 0; i < width/2; i++) {
            pts_left_list.add(new Point(pts_left.get(i, 0)[0], pts_left.get(i, 1)[0]));
            pts_right_list.add(new Point(pts_right.get(i, 0)[0], pts_right.get(i, 1)[0]));
        }

        MatOfPoint matOfPointLeft = new MatOfPoint();
        matOfPointLeft.fromList(pts_left_list);
        MatOfPoint matOfPointRight = new MatOfPoint();
        matOfPointRight.fromList(pts_right_list);

        double[] color = {0, 255, 0};

        List<MatOfPoint> listOfPoints = new ArrayList<>();
        listOfPoints.add(matOfPointLeft);
        Imgproc.polylines(color_warp, listOfPoints, false, new Scalar(color), 5);

        listOfPoints.clear();
        listOfPoints.add(matOfPointRight);
        Imgproc.polylines(color_warp, listOfPoints, false, new Scalar(color), 5);

        Mat newwarp = new Mat();
        Imgproc.warpPerspective(color_warp, newwarp, Minv, new Size(width, height), Imgproc.INTER_LINEAR);

        result = new Mat();

        Core.addWeighted(img, 1, newwarp, 0.5, 0, result);
    }
}
