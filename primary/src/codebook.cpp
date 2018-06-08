#include "codebook.h"
BackgroundSubtractorCodeBook::BackgroundSubtractorCodeBook()
{
   nChannels = CHANNELS;
}

//初始化maskImage,codebook数组，将每个像素的codebook的码元个数置0，初始化建模阈值和分割阈值
void BackgroundSubtractorCodeBook::initialize(const Mat &inputRGBImage,Mat &outputImage)
{
    if (inputRGBImage.empty())
    {
        return ;
    }
    if (yuvImage.empty())
    {
        yuvImage.create(inputRGBImage.size(),inputRGBImage.type());
    }
    if (maskImage.empty())
    {
        maskImage.create(inputRGBImage.size(),CV_8UC1);
        Mat temp(inputRGBImage.rows,inputRGBImage.cols,CV_8UC1,Scalar::all(255));
        maskImage=temp;

    }
    imageSize=inputRGBImage.cols*inputRGBImage.rows;


    codebookVec=new CodeBook[imageSize];
    for (int i=0;i<imageSize;++i)
    {
        codebookVec[i].numEntries=0;
    }//初始化码元个数
    for (int i=0; i<nChannels; i++)
    {
        cbBounds[i] = 10;    // 用于确定码元各通道的建模阀值
        minMod[i]    = 20;    // 用于背景差分函数中
        maxMod[i]    = 20;    // 调整其值以达到最好的分割
    }
    outputImage=maskImage;
}

//逐个像素建模
void BackgroundSubtractorCodeBook::updateCodeBook(const Mat &inputImage)
{
    cvtColor(inputImage,yuvImage,CV_RGB2YCrCb);
    pColor=yuvImage.data;
    for (int c=0;c<imageSize;++c)
    {
        _updateCodeBookPerPixel(c);
        pColor+=3;
    }
}

//单个像素建模实现函数，遍历所有码元，分三通道匹配，若满足，则更新该码元的时间，最大，最小值
//若不匹配，则创建新的码元。
void BackgroundSubtractorCodeBook::_updateCodeBookPerPixel(int pixelIndex)
{
    if (codebookVec[pixelIndex].numEntries==0)
    {
        codebookVec[pixelIndex].t=0;
    }
    codebookVec[pixelIndex].t+=1;


    int n;
    unsigned int high[3],low[3];
    for (n=0; n<nChannels; n++)    //处理三通道的三个像素
    {
        high[n] = *(pColor+n) + *(cbBounds+n);
        // *(p+n) 和 p[n] 结果等价,经试验*(p+n) 速度更快
        if(high[n] > 255) high[n] = 255;
        low[n] = *(pColor+n)-*(cbBounds+n);
        if(low[n] < 0) low[n] = 0;
        // 用p 所指像素通道数据,加减cbBonds中数值,作为此像素阀值的上下限
    }

    int matchChannel;

    list<CodeWord>::iterator jList;
    list<CodeWord>::iterator jListAfterPush;

    for (jList=codebookVec[pixelIndex].codeElement.begin();jList!=codebookVec[pixelIndex].codeElement.end();++jList)
    {
        // 遍历此码本每个码元,测试p像素是否满足其中之一
        matchChannel = 0;
        for (n=0; n<nChannels; n++)
            //遍历每个通道
        {
            if(((*jList).learnLow[n]<= *(pColor+n))
                && (*(pColor+n) <= (*jList).learnHigh[n])) //Found an entry for this channel
                // 如果p 像素通道数据在该码元阀值上下限之间
            {
                matchChannel++;
            }
        }
        if (matchChannel == nChannels)        // If an entry was found over all channels
            // 如果p 像素各通道都满足上面条件
        {
            (*jList).t_last_update = codebookVec[pixelIndex].t;
            // 更新该码元时间为当前时间
            // adjust this codeword for the first channel
            for (n=0; n<nChannels; n++)
                //调整该码元各通道最大最小值
            {
                if ((*jList).max[n] < *(pColor+n))
                    (*jList).max[n] = *(pColor+n);
                else if ((*jList).min[n] > *(pColor+n))
                    (*jList).min[n] = *(pColor+n);
            }
            break;//如果满足其中一个码元，则退出循环。
        }
    }

    // ENTER A NEW CODE WORD IF NEEDED
    if(jList==codebookVec[pixelIndex].codeElement.end())  // No existing code word found, make a new one
        // p 像素不满足此码本中任何一个码元,下面创建一个新码元
    {
        CodeWord newElement;
        for(n=0; n<nChannels; n++)
            // 更新新码元各通道数据
        {
            newElement.learnHigh[n] = high[n];
            newElement.learnLow[n] = low[n];
            newElement.max[n] = *(pColor+n);
            newElement.min[n] = *(pColor+n);
        }
        newElement.t_last_update = codebookVec[pixelIndex].t;
        newElement.stale = 0;
        codebookVec[pixelIndex].numEntries += 1;
        codebookVec[pixelIndex].codeElement.push_back(newElement);//新的码元加入链表的最后。
    }

    // OVERHEAD TO TRACK POTENTIAL STALE ENTRIES
    for (jListAfterPush=codebookVec[pixelIndex].codeElement.begin();
        jListAfterPush!=codebookVec[pixelIndex].codeElement.end();++jListAfterPush)
    {
        // This garbage is to track which codebook entries are going stale
        int negRun = codebookVec[pixelIndex].t - (*jListAfterPush).t_last_update;
        // 计算该码元的不更新时间
        if((*jListAfterPush).stale < negRun)
            (*jListAfterPush).stale = negRun;
    }

    // SLOWLY ADJUST LEARNING BOUNDS
    //对符合的码元进行更新，刚建立的码元肯定不满足条件，不用考虑
    for(n=0; n<nChannels; n++)
        // 如果像素通道数据在高低阀值范围内,但在码元阀值之外,则缓慢调整此码元学习界限
    {
            if((*jList).learnHigh[n] < high[n])
                (*jList).learnHigh[n] += 1;//+1是什么意思？缓慢调整？
            if((*jList).learnLow[n] > low[n])
                (*jList).learnLow[n] -= 1;
    }

    return;
}

void BackgroundSubtractorCodeBook::clearStaleEntries()
{
    for (int i=0;i<imageSize;++i)
    {
        _clearStaleEntriesPerPixel(i);
    }
}

void BackgroundSubtractorCodeBook::_clearStaleEntriesPerPixel(int pixelIndex)
{
    int staleThresh=codebookVec[pixelIndex].t;
    for (list<CodeWord>::iterator itor=codebookVec[pixelIndex].codeElement.begin();
        itor!=codebookVec[pixelIndex].codeElement.end();)
    {
        if ((*itor).stale>staleThresh)
        {
            itor=codebookVec[pixelIndex].codeElement.erase(itor);//erase之后返回被删的下一个元素的位置
        }
        else
        {
            (*itor).stale=0;
            (*itor).t_last_update=0;
            itor++;
        }
    }
    codebookVec[pixelIndex].t=0;//码本时间清零；

    codebookVec[pixelIndex].numEntries=(int)codebookVec[pixelIndex].codeElement.size();

    return;

}

void BackgroundSubtractorCodeBook:: backgroudDiff(const Mat &inputImage,Mat &outputImage)
{
    cvtColor(inputImage,yuvImage,CV_RGB2YCrCb);
    pColor=yuvImage.data;
    pMask =maskImage.data;
    for(int c=0; c<imageSize; c++)
    {
        maskPixelCodeBook = _backgroundDiff(c);
        *pMask++ = maskPixelCodeBook;
        pColor += 3;
        // pColor 指向的是3通道图像
    }
    outputImage=maskImage.clone();
}

uchar BackgroundSubtractorCodeBook::_backgroundDiff(int pixelIndex)
{
    int matchChannels;
    list<CodeWord>::iterator itor;
    for (itor=codebookVec[pixelIndex].codeElement.begin();
        itor!=codebookVec[pixelIndex].codeElement.end();++itor)
    {
        matchChannels=0;
        for (int n=0;n<nChannels;++n)
        {

            if (((*itor).min[n] - minMod[n] <= *(pColor+n)) && (*(pColor+n) <=(*itor).max[n] + maxMod[n]))
                //相对于背景学习，这里是与码元中的最大最小值比较，并加入了余量minMod,maxMod;
                matchChannels++; //Found an entry for this channel
            else
                break;//一个通道没匹配，直接退出
        }
        if (matchChannels == nChannels)
            break; //Found an entry that matched all channels,确定是背景像素 返回0 黑色
    }
    if (itor==codebookVec[pixelIndex].codeElement.end())
    {
        return(255);
    }
    return (0);
}

BackgroundSubtractorCodeBook::~BackgroundSubtractorCodeBook()
{
    delete [] codebookVec;
}
