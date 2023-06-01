#include "tdataunit.hpp"

#include "modelparameter.h"

TdataUnit* DataUnit = new TdataUnit();
TdataUnit::TdataUnit()
{
    this->strategyname = 8;
    this->HSVColorRange = new ColorRange* [DefLabelMarkSize];
    for(int ColorRangeCnt = 0; ColorRangeCnt < DefLabelMarkSize; ColorRangeCnt++)
    {
        this->HSVColorRange[ColorRangeCnt] = new ColorRange;
        this->HSVColorRange[ColorRangeCnt]->HueMax = 1.0;
        this->HSVColorRange[ColorRangeCnt]->HueMin = 0.0;
        this->HSVColorRange[ColorRangeCnt]->SaturationMax = 1.0;
        this->HSVColorRange[ColorRangeCnt]->SaturationMin = 0.0;
        this->HSVColorRange[ColorRangeCnt]->BrightnessMax = 1.0;
        this->HSVColorRange[ColorRangeCnt]->BrightnessMin = 0.0;
        switch (ColorRangeCnt) {
        case 0:
            this->HSVColorRange[ColorRangeCnt]->LabelName = "[Orange Range]";
            break;
        case 1:
            this->HSVColorRange[ColorRangeCnt]->LabelName = "[Yellow Range]";
            break;
        case 2:
            this->HSVColorRange[ColorRangeCnt]->LabelName = "[Blue Range]";
            break;
        case 3:
            this->HSVColorRange[ColorRangeCnt]->LabelName = "[Green Range]";
            break;
        case 4:
            this->HSVColorRange[ColorRangeCnt]->LabelName = "[Black Range]";
            break;
        case 5:
            this->HSVColorRange[ColorRangeCnt]->LabelName = "[Red Range]";
            break;
        case 6:
            this->HSVColorRange[ColorRangeCnt]->LabelName = "[White Range]";
            break;
        case 7:
            this->HSVColorRange[ColorRangeCnt]->LabelName = "[Other Range]";
            break;
        default:
            break;
        }
    }

}

TdataUnit::~TdataUnit()
{
    for(int ColorRangeCnt = 0; ColorRangeCnt < DefLabelMarkSize; ColorRangeCnt++)
    {
        delete [] this->HSVColorRange[ColorRangeCnt];
    }
    delete [] this->HSVColorRange;
}

void TdataUnit::SaveColorRangeFile(std::string location)
{
    char path[200];
    strcpy(path, tool->getPackagePath("strategy").c_str());
    strcat(path, "/");
    strcat(path, location.c_str());
    strcat(path, "/Parameter/ColorModelData.ini");
    try
    {
//        ofstream OutFile(sFileName.c_str());
        ofstream OutFile(path);
        for(int ColorRangeCnt = 0; ColorRangeCnt < DefLabelMarkSize; ColorRangeCnt++)
        {
            OutFile << this->HSVColorRange[ColorRangeCnt]->LabelName;
            OutFile << "\n";
            OutFile << "Hue_Max = ";
            OutFile << this->HSVColorRange[ColorRangeCnt]->HueMax;
            OutFile << "\n";
            OutFile << "Hue_Min = ";
            OutFile << this->HSVColorRange[ColorRangeCnt]->HueMin;
            OutFile << "\n";
            OutFile << "Saturation_Max = ";
            OutFile << this->HSVColorRange[ColorRangeCnt]->SaturationMax;
            OutFile << "\n";
            OutFile << "Saturation_Min = ";
            OutFile << this->HSVColorRange[ColorRangeCnt]->SaturationMin;
            OutFile << "\n";
            OutFile << "Brightness_Max = ";
            OutFile << this->HSVColorRange[ColorRangeCnt]->BrightnessMax;
            OutFile << "\n";
            OutFile << "Brightness_Min = ";
            OutFile << this->HSVColorRange[ColorRangeCnt]->BrightnessMin;
            OutFile << "\n";
        }
        OutFile.close();
    }
    catch( exception e )
    {
    }
}

void TdataUnit::LoadColorRangeFile(std::string location)
{
    fstream fin;
    char line[100]; 
    char path[200];
    strcpy(path, tool->getPackagePath("strategy").c_str());
    printf("location:%s\n",location.c_str());
    strcat(path, "/");
    strcat(path, location.c_str());
    strcat(path, "/Parameter/ColorModelData.ini");
    printf("%s\n",path);
    fin.open(path, ios::in);
    //fin.open(("../../Parameter/Color_Model_Data/ColorModelData.ini"), ios::in);
    try
    {
        for(int ColorCnt = 0; ColorCnt < 8; ColorCnt++)
        {
            fin.getline(line,sizeof(line),'\n');
            HSVColorRange[ColorCnt]->HueMax = tool->readvalue(fin, "Hue_Max", 1);
            HSVColorRange[ColorCnt]->HueMin = tool->readvalue(fin, "Hue_Min", 1);
            HSVColorRange[ColorCnt]->SaturationMax = tool->readvalue(fin, "Saturation_Max", 1);
            HSVColorRange[ColorCnt]->SaturationMin = tool->readvalue(fin, "Saturation_Min", 1);
            HSVColorRange[ColorCnt]->BrightnessMax = tool->readvalue(fin, "Brightness_Max", 1);
            HSVColorRange[ColorCnt]->BrightnessMin = tool->readvalue(fin, "Brightness_Min", 1);
        }
        fin.close();
    }
    catch(exception e)
    {
        printf("can't open");
    }
}


