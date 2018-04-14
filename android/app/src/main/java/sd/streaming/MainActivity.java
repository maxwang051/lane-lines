package sd.streaming;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.ImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.Timer;
import java.util.TimerTask;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "OCVSample::Activity";

    int count;
    ImageView imageView;

    Mat img, warpSrc, warpDst, M, Minv, warped, gray, hls, lab;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    img = new Mat();
                    warpSrc = new Mat();
                    warpSrc.create(4, 2, CvType.CV_32F);

                    warpDst = new Mat();
                    warpDst.create(4, 2, CvType.CV_32F);

                    M = new Mat();
                    M.create(3,3, CvType.CV_32F);
                    Minv = new Mat();
                    warped = new Mat();

                    gray = new Mat();
                    hls = new Mat();
                    lab = new Mat();
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

    }

    void nextImage() {
        String packageName = getPackageName();
        String type = "drawable";
        int id = getResources().getIdentifier("image" + Integer.toString(count),
                type, packageName);
        imageView = (ImageView) findViewById(R.id.imageView);

        Bitmap bmp = BitmapFactory.decodeResource(getResources(),R.drawable.image0);
        //Mat source = new Mat();
        try {
            img = Utils.loadResource(MainActivity.this, id);
            Imgproc.cvtColor(img, img, Imgproc.COLOR_RGB2BGRA);
        } catch (IOException e) {
            Log.e(TAG,Log.getStackTraceString(e));
        }

        warped.create(img.cols(), img.rows(), CvType.CV_32F);
        gray.create(img.cols(), img.rows(), CvType.CV_32F);
        hls.create(img.cols(), img.rows(), CvType.CV_32F);
        lab.create(img.cols(), img.rows(), CvType.CV_32F);

        processImage(img);

        Bitmap bm = Bitmap.createBitmap(gray.cols(), gray.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(gray, bm);
        imageView.setImageBitmap(bm);

        count = (count + 1) % 64;
    }

    void processImage(Mat img) {
        int height = img.height();
        int width = img.width();

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

        Imgproc.cvtColor(warped, gray, Imgproc.COLOR_RGB2GRAY);
        Imgproc.cvtColor(warped, hls, Imgproc.COLOR_RGB2HLS);
        Imgproc.cvtColor(warped, lab, Imgproc.COLOR_RGB2Lab);

    }

}