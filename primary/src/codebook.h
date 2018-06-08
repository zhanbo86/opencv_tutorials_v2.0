#ifndef CODEBOOK_H
#define CODEBOOK_H

#include <iostream>
#include <list>
#include <opencv2/opencv.hpp>

#define CHANNELS 3

using namespace std;
using namespace cv;

class CodeWord
{
public:
    uchar    learnHigh[CHANNELS];    // High side threshold for learning
    // 此码元各通道的阀值上限(背景学习建模界限)
    uchar    learnLow[CHANNELS];        // Low side threshold for learning
    // 此码元各通道的阀值下限（）
    // 学习过程中如果一个新像素各通道值x[i],均有 learnLow[i]<=x[i]<=learnHigh[i],则该像素可合并于此码元
    uchar    max[CHANNELS];            // High side of box boundary  实际最大值
    // 属于此码元的像素中各通道的最大值（判断过程的界限）
    uchar    min[CHANNELS];            // Low side of box boundary   实际最小值
    // 属于此码元的像素中各通道的最小值
    int        t_last_update;            // This is book keeping to allow us to kill stale entries
    // 此码元最后一次更新的时间,每一帧为一个单位时间,用于计算stale
    int        stale;                    // max negative run (biggest period of inactivity)
    // 此码元最长不更新时间,用于删除规定时间不更新的码元,精简码本

};

class CodeBook
{
public:
    list<CodeWord> codeElement;
    //考虑到方便的删除旧码元节点，同时访问码元的时候只需要顺序遍历，选用STL中的List,避免用指针。
    int                numEntries;
    // 此码本中码元的数目
    int                t;                // count every access
    // 此码本现在的时间,一帧为一个时间单位                    // 码本的数据结构
};


class BackgroundSubtractorCodeBook
{
public:

    BackgroundSubtractorCodeBook();
    void updateCodeBook(const Mat &inputImage);//背景建模
    void initialize(const Mat &inputImagee,Mat &outputImage);//获取第一帧，进行初始化
    void clearStaleEntries();//清除旧的码元
    void backgroudDiff(const Mat &inputImage,Mat &outputImage);//背景前景判断
    ~BackgroundSubtractorCodeBook();

private:

    Mat         yuvImage;
    Mat         maskImage;
    CodeBook*   codebookVec;    //本来想用Vector，不用数组指针，但是codebook定长，不用增减，vector也不好初始化。
    unsigned    cbBounds[CHANNELS]; //背景建模时，用于建立学习的上下界限
    uchar*        pColor; //yuvImage pointer  通道指针
    uchar*      pMask;// maskImage pointer
    int            imageSize;
    int            nChannels;
    int            minMod[CHANNELS];  //判断背景或者前景所用调节阈值
    int            maxMod[CHANNELS];


    uchar maskPixelCodeBook;
    void _updateCodeBookPerPixel(int pixelIndex);
    void _clearStaleEntriesPerPixel(int pixelIndex);
    uchar _backgroundDiff(int pixelIndex);
};


#endif // CODEBOOK_H
