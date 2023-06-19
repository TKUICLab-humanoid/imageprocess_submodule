#include "imagemain.h"

void ModelingFunction(const tku_msgs::ButtonColorForm& msg)
{
    if(msg.BuildingModel)
    {
        ModelUnit->isBuildModel = true;
    }
    else
    {
        ModelUnit->isBuildModel = false;
    }
}
void ChangeHSVValue(const tku_msgs::HSVValue& msg)
{
    ModelUnit->hsvColorRange->HueMax = (float)msg.HMax/HueScrollBarMax;
    ModelUnit->hsvColorRange->HueMin = (float)msg.HMin/HueScrollBarMax;
    ModelUnit->hsvColorRange->SaturationMax = (float)msg.SMax/SaturationScrollBarMax;
    ModelUnit->hsvColorRange->SaturationMin = (float)msg.SMin/SaturationScrollBarMax;
    ModelUnit->hsvColorRange->BrightnessMax = (float)msg.VMax/BrightnessScrollBarMax;
    ModelUnit->hsvColorRange->BrightnessMin = (float)msg.VMin/BrightnessScrollBarMax;
}
void DrawImageFunction(const tku_msgs::DrawImage& msg)
{
    
    if(dra.isnewdraw(msg.cnt,drawImageArray))
    {
    //create a new space to save drawing data
        drawImageInfo tmp;
        tmp.cnt = msg.cnt;
        tmp.XMax = msg.XMax;
        tmp.XMin = msg.XMin;
        tmp.YMax = msg.YMax;
        tmp.YMin = msg.YMin;
        tmp.rValue = msg.rValue;
        tmp.gValue = msg.gValue;
        tmp.bValue = msg.bValue;
        tmp.Mode = msg.Mode;
        drawImageArray.push_back(tmp);
    }
    else
    {
    //find out which cnt is this time want draw and change data inside
        for(int i=0 ;i<drawImageArray.size();i++){
            if(drawImageArray[i].cnt==msg.cnt){
                drawImageArray[i].XMax = msg.XMax;
                drawImageArray[i].XMin = msg.XMin;
                drawImageArray[i].YMax = msg.YMax;
                drawImageArray[i].YMin = msg.YMin;
                drawImageArray[i].rValue = msg.rValue;
                drawImageArray[i].gValue = msg.gValue;
                drawImageArray[i].bValue = msg.bValue;
            }
        }
    }
}

bool LoadHSVInfo(tku_msgs::HSVInfo::Request &HSVreq, tku_msgs::HSVInfo::Response &HSVres)
{
    ModelUnit->hsvColorRange = DataUnit->HSVColorRange[HSVreq.ColorLabel];
    ModelUnit->ColorSelected = HSVreq.ColorLabel;
    HSVres.Hmax = DataUnit->HSVColorRange[HSVreq.ColorLabel]->HueMax * HueScrollBarMax;
    HSVres.Hmin = DataUnit->HSVColorRange[HSVreq.ColorLabel]->HueMin * HueScrollBarMax;
    HSVres.Smax = DataUnit->HSVColorRange[HSVreq.ColorLabel]->SaturationMax * SaturationScrollBarMax;
    HSVres.Smin = DataUnit->HSVColorRange[HSVreq.ColorLabel]->SaturationMin * SaturationScrollBarMax;
    HSVres.Vmax = DataUnit->HSVColorRange[HSVreq.ColorLabel]->BrightnessMax * BrightnessScrollBarMax;
    HSVres.Vmin = DataUnit->HSVColorRange[HSVreq.ColorLabel]->BrightnessMin * BrightnessScrollBarMax;
    return true;
}
bool CallBuildFunction(tku_msgs::BuildModel::Request &req, tku_msgs::BuildModel::Response &res)
{
    if(req.Build)
    {
        ModelUnit->HSV_BuildingColorModel();
        res.Already = true;
    }
}
bool CallSaveHSVFunction(tku_msgs::SaveHSV::Request &req, tku_msgs::SaveHSV::Response &res)
{
    if(req.Save)
    {
        DataUnit->SaveColorRangeFile(req.location);
        res.Already = true;
    }
}

void GetImagesourceFunction(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    checkImageSource=true;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      buffer = cv_ptr->image;
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
}

void Catch_image(const sensor_msgs::ImageConstPtr& msg)
{
    if(!checkRealImage)
    {
        cv_bridge::CvImagePtr cv_ptr;
        checkImageSource=true;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        buffer = cv_ptr->image;
    }
    else
    {
        ROS_ERROR("Please Close Real Image and restart the node.");
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "imageprocess");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    std::string location;
    nh.getParam("/location",location);
    tku_msgs::SaveHSV::Request request;
    request.location = location;

    Object_Publish = nh.advertise<tku_msgs::ObjectList>("/Object/List", 1);
    LabelModel_Publish = nh.advertise<tku_msgs::LabelModelObjectList>("/LabelModel/List", 1);

    ModelingButton_subscriber = nh.subscribe("ColorModelForm_Topic", 1000, ModelingFunction);
    HSVValue_subscriber = nh.subscribe("HSVValue_Topic", 1000, ChangeHSVValue);
    DrawImage_subscriber = nh.subscribe("/strategy/drawimage", 100, DrawImageFunction);
    Imagesource_subscriber = nh.subscribe("/usb_cam/image_raw", 10, GetImagesourceFunction);
    Gazeboimage_subscriber = nh.subscribe("/kidsize/camera/image_raw", 10, Catch_image);

    HSV_service = nh.advertiseService("LoadHSVInfo", LoadHSVInfo);
    Build_service = nh.advertiseService("BuildModel", CallBuildFunction);
    Save_service = nh.advertiseService("SaveHSV", CallSaveHSVFunction);
    
    pub_orign = it.advertise("orign_image", 1);
    pub_colormodel = it.advertise("colormodel_image", 1);
    pub_compressimage = it.advertise("compress_image", 1);

    FPScnt = 0;
    drawImagecnt = 0;
    drawImageArray.clear();
    
    ModelUnit->HSV_BuildingTable();
    printf("\nloadColor\n");
    DataUnit->LoadColorRangeFile(location);
    ModelUnit->HSV_BuildingColorModel();

    ros::spinOnce();

    ros::Rate loop_rate(30);

   while (nh.ok())
    {
        if(!buffer.empty()&&checkImageSource)
        {
            FPScnt++;
            cv::resize(buffer, orign, cv::Size(320, 240));

            colormodel = orign.clone();
            if(ModelUnit->isBuildModel)
            {
                ModelUnit->ColorModeling(colormodel);
            }
            else
            {
                ModelUnit->ChangeToColorModel(colormodel);
                ImageBase->TErosionMain(ModelUnit->Label_Model);
                ImageBase->TDilation(ModelUnit->Label_Model);
                ModelUnit->ChangeToColorModel_ImageProcess(colormodel);
/////////////////////////////////////////////////////compress_image/////////////////////////////////////////////////////
                cv::resize(colormodel, compress_image, cv::Size(32, 24));
                msg_compressimage = cv_bridge::CvImage(std_msgs::Header(), "bgr8", compress_image).toImageMsg();
                pub_compressimage.publish(msg_compressimage);
/////////////////////////////////////////////////////compress_image/////////////////////////////////////////////////////
                ImageBase->TSegmentationMain(ModelUnit->Label_Model);
                for(int i = 0; i < drawImageArray.size(); i++)
                {
                    switch (drawImageArray[i].Mode) {
                    case etDrawLine:
                        line(orign,Point(drawImageArray[i].XMin,drawImageArray[i].YMin),Point(drawImageArray[i].XMax,drawImageArray[i].YMax),
                             Scalar(drawImageArray[i].bValue,drawImageArray[i].gValue,drawImageArray[i].rValue),thickness,linetype);
                        break;
                    case etDrawObject:
                        rectangle(orign,Point(drawImageArray[i].XMin,drawImageArray[i].YMin),Point(drawImageArray[i].XMax,drawImageArray[i].YMax),
                                  Scalar(drawImageArray[i].bValue,drawImageArray[i].gValue,drawImageArray[i].rValue),thickness,linetype);
                        break;
                    }
                }
                for(int i = 0; i < DefLabelMarkSize; i++)
                {
                    for(int j = 0; j < ImageBase->ObjectList[i]->ObjectCnt; j++)
                    {
                        colordata.X = ImageBase->ObjectList[i]->ObjectInfoList[j].Center.X;
                        colordata.Y = ImageBase->ObjectList[i]->ObjectInfoList[j].Center.Y;
                        colordata.XMin = ImageBase->ObjectList[i]->ObjectInfoList[j].XMin;
                        colordata.YMin = ImageBase->ObjectList[i]->ObjectInfoList[j].YMin;
                        colordata.XMax = ImageBase->ObjectList[i]->ObjectInfoList[j].XMax;
                        colordata.YMax = ImageBase->ObjectList[i]->ObjectInfoList[j].YMax;
                        colordata.Width = ImageBase->ObjectList[i]->ObjectInfoList[j].ObjectWidth;
                        colordata.Height = ImageBase->ObjectList[i]->ObjectInfoList[j].ObjectHigh;
                        colordata.size = ImageBase->ObjectList[i]->ObjectInfoList[j].AreaSize;
                        colorarray.Colorarray.push_back(colordata);
                    }
                    colorarray.cnt = ImageBase->ObjectList[i]->ObjectCnt;
                    objectlist.Objectlist.push_back(colorarray);
                    colorarray.Colorarray.clear();
                }
                Object_Publish.publish(objectlist);
                LabelModel_Publish.publish(ModelUnit->labelmodel);
                ModelUnit->labelmodel.LabelModel.clear();
                objectlist.Objectlist.clear();
            }

            msg_orign = cv_bridge::CvImage(std_msgs::Header(), "bgr8", orign).toImageMsg();
            msg_colormodel = cv_bridge::CvImage(std_msgs::Header(), "bgr8", colormodel).toImageMsg();
            pub_orign.publish(msg_orign);
            pub_colormodel.publish(msg_colormodel);

        }
        checkImageSource=false;
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}



