#include "imagebase.hpp"

TImageBase* ImageBase = new TImageBase();

TImageBase::TImageBase()
{
    this->ImageSize = IMAGEHEIGHT*IMAGEWIDTH*3;
    this->bNewFlag = false;
    this->tmpModelBuffer = new unsigned char[IMAGEHEIGHT*IMAGEWIDTH]();
    for( int LabelCnt = 0; LabelCnt < 8 ; LabelCnt++ ){
        this->ObjectList.push_back( new TImageInfo() );//ÀxŠs€£ŠPÃCŠâªºª«¥ó®eŸ¹
    }
    this->ProcessLabelValue = 0xff;
}

void TImageBase::TDilation(unsigned char* ModelBuffer)
{
    unsigned char tmp;
    for(int iHeightCnt = 1; iHeightCnt < IMAGEHEIGHT - 1; iHeightCnt++)
    {
        for(int iWidthCnt = 1; iWidthCnt < IMAGEWIDTH - 1; iWidthCnt++)
        {
            tmp = (ModelBuffer[(iHeightCnt - 1) * IMAGEWIDTH + (iWidthCnt - 1)] | ModelBuffer[(iHeightCnt - 1) * IMAGEWIDTH + (iWidthCnt)] | ModelBuffer[(iHeightCnt - 1) * IMAGEWIDTH + (iWidthCnt + 1)] |
                    ModelBuffer[(iHeightCnt) * IMAGEWIDTH + (iWidthCnt - 1)] | ModelBuffer[(iHeightCnt) * IMAGEWIDTH + (iWidthCnt)] | ModelBuffer[(iHeightCnt) * IMAGEWIDTH + (iWidthCnt + 1)] |
                    ModelBuffer[(iHeightCnt + 1) * IMAGEWIDTH + (iWidthCnt - 1)] | ModelBuffer[(iHeightCnt + 1) * IMAGEWIDTH + (iWidthCnt)] | ModelBuffer[(iHeightCnt + 1) * IMAGEWIDTH + (iWidthCnt + 1)]);
            if(tmp == OrangeLabel || tmp == YellowLabel || tmp == BlueLabel || tmp == GreenLabel || tmp == BlackLabel || tmp == RedLabel || tmp == WhiteLabel || tmp == OthersLabel)
            {
                this->tmpModelBuffer[(iHeightCnt * IMAGEWIDTH) + iWidthCnt] = tmp;
            }
            else
            {
                this->tmpModelBuffer[(iHeightCnt * IMAGEWIDTH) + iWidthCnt] = ModelBuffer[(iHeightCnt * IMAGEWIDTH) + iWidthCnt];
            }
        }
    }
    memcpy(ModelBuffer, this->tmpModelBuffer, IMAGEHEIGHT * IMAGEWIDTH);
}

void TImageBase::TErosionMain(unsigned char* ModelBuffer)
{
    for(int iHeightCnt = 1; iHeightCnt < IMAGEHEIGHT - 1; iHeightCnt++)
    {
        for(int iWidthCnt = 1; iWidthCnt < IMAGEWIDTH - 1; iWidthCnt++)
        {
            if(ModelBuffer[(iHeightCnt - 1) * IMAGEWIDTH + (iWidthCnt - 1)] & ModelBuffer[(iHeightCnt - 1) * IMAGEWIDTH + (iWidthCnt)] & ModelBuffer[(iHeightCnt - 1) * IMAGEWIDTH + (iWidthCnt + 1)] &
                    ModelBuffer[(iHeightCnt) * IMAGEWIDTH + (iWidthCnt - 1)] & ModelBuffer[(iHeightCnt) * IMAGEWIDTH + (iWidthCnt)] & ModelBuffer[(iHeightCnt) * IMAGEWIDTH + (iWidthCnt + 1)] &
                    ModelBuffer[(iHeightCnt + 1) * IMAGEWIDTH + (iWidthCnt - 1)] & ModelBuffer[(iHeightCnt + 1) * IMAGEWIDTH + (iWidthCnt)] & ModelBuffer[(iHeightCnt + 1) * IMAGEWIDTH + (iWidthCnt + 1)] != ModelBuffer[iHeightCnt * IMAGEWIDTH + iWidthCnt])
            {
                this->tmpModelBuffer[(iHeightCnt * IMAGEWIDTH) + iWidthCnt] = 0x00;
            }
            else
            {
                this->tmpModelBuffer[(iHeightCnt * IMAGEWIDTH) + iWidthCnt] = ModelBuffer[(iHeightCnt * IMAGEWIDTH) + iWidthCnt];
            }
        }
    }
    memcpy(ModelBuffer, this->tmpModelBuffer, IMAGEHEIGHT * IMAGEWIDTH);
}

void TImageBase::Boundary(TObjectInfo &ObjectInfo, TCoordinate Coordinate)
{
    if( ObjectInfo.XMax < Coordinate.X ){
        ObjectInfo.XMax = Coordinate.X;
    }else if( ObjectInfo.XMin > Coordinate.X ){
        ObjectInfo.XMin = Coordinate.X;
    }

    if( ObjectInfo.YMax < Coordinate.Y ){
        ObjectInfo.YMax = Coordinate.Y;
    }else if( ObjectInfo.YMin > Coordinate.Y ){
        ObjectInfo.YMin = Coordinate.Y;
    }
}

void TImageBase::ClearAllObject()
{
    for( int LabelCnt = 0; LabelCnt < 8 && LabelCnt < ObjectList.size(); LabelCnt++ ){
        this->ObjectList[LabelCnt]->ObjectCnt = 0;
        this->ObjectList[LabelCnt]->ObjectInfoList.clear();
    }
}

bool TImageBase::TSegmentationMain(unsigned char *modelbuffer)
{
    this->bNewFlag = false;
    ClearAllObject();
    int Index;
    int iHeightOffset = 0,iWidthOffset = 0;

    unsigned char *ModelBuffer = new unsigned char[ImageSize];
    memcpy(ModelBuffer, modelbuffer, ImageSize);
    for( int iHeightCnt = 0; iHeightCnt < IMAGEHEIGHT - 1; iHeightCnt++ ){
        for( int iWidthCnt = 0; iWidthCnt < IMAGEWIDTH - 1; iWidthCnt++ ){

            if ( ModelBuffer[( iHeightCnt ) * IMAGEWIDTH + iWidthCnt ] == 0 )
                continue;


            Index = 0;

            TObjectInfo ObjectInfo( iWidthCnt , iHeightCnt );//Object
            ObjectInfo.MaskLabel = ModelBuffer[( iHeightCnt ) * IMAGEWIDTH + iWidthCnt ];//mark€§ÃCŠâ

            ModelBuffer[( iHeightCnt ) * IMAGEWIDTH + iWidthCnt ] &= 0x00;

            while( Index <= ObjectInfo.CoordinateCnt - 1 ){
                if( ObjectInfo.CoordinateList[Index].Y != iHeightOffset && ObjectInfo.CoordinateList[Index].X != iWidthOffset &&
                        ObjectInfo.CoordinateList[Index].Y != ( IMAGEHEIGHT + iHeightOffset - 1 ) &&  ObjectInfo.CoordinateList[Index].X != ( IMAGEWIDTH + iWidthOffset - 1 ) ){//€£¹ïÃä®Ø§@ª«¥ó€À³Î

                    if( ModelBuffer[( ( ObjectInfo.CoordinateList[Index].Y - 1 ) * IMAGEWIDTH + ( ObjectInfo.CoordinateList[Index].X - 1 ) )] & ObjectInfo.MaskLabel ){//Š³mark
                        ObjectInfo.CoordinateCnt++;
                        TCoordinate Coordinate( ObjectInfo.CoordinateList[Index].X - 1, ObjectInfo.CoordinateList[Index].Y - 1 );
                        ObjectInfo.CoordinateList.push_back( Coordinate );
                        ModelBuffer[( ( ObjectInfo.CoordinateList[Index].Y - 1 ) * IMAGEWIDTH + ( ObjectInfo.CoordinateList[Index].X - 1 ) )] &= 0x00;//šú®øMark
                        this->Boundary( ObjectInfo, Coordinate );
                    }

                    if( ModelBuffer[( ( ObjectInfo.CoordinateList[Index].Y - 1 ) * IMAGEWIDTH + ObjectInfo.CoordinateList[Index].X )] & ObjectInfo.MaskLabel ){
                        ObjectInfo.CoordinateCnt++;
                        TCoordinate Coordinate( ObjectInfo.CoordinateList[Index].X, ObjectInfo.CoordinateList[Index].Y - 1 );
                        ObjectInfo.CoordinateList.push_back( Coordinate );
                        ModelBuffer[( ( ObjectInfo.CoordinateList[Index].Y - 1 ) * IMAGEWIDTH + ObjectInfo.CoordinateList[Index].X )] &= 0x00;//šú®øMark
                        this->Boundary( ObjectInfo, Coordinate );
                    }

                    if( ModelBuffer[( ( ObjectInfo.CoordinateList[Index].Y - 1 ) * IMAGEWIDTH + ObjectInfo.CoordinateList[Index].X + 1 )] & ObjectInfo.MaskLabel ){
                        ObjectInfo.CoordinateCnt++;
                        TCoordinate Coordinate( ObjectInfo.CoordinateList[Index].X + 1, ObjectInfo.CoordinateList[Index].Y - 1 );
                        ObjectInfo.CoordinateList.push_back( Coordinate );
                        ModelBuffer[( ( ObjectInfo.CoordinateList[Index].Y - 1 ) * IMAGEWIDTH + ObjectInfo.CoordinateList[Index].X + 1 )] &= 0x00;//šú®øMark
                        this->Boundary( ObjectInfo, Coordinate );
                    }

                    if( ModelBuffer[( ObjectInfo.CoordinateList[Index].Y * IMAGEWIDTH + ObjectInfo.CoordinateList[Index].X + 1 )] & ObjectInfo.MaskLabel ){
                        ObjectInfo.CoordinateCnt++;
                        TCoordinate Coordinate( ObjectInfo.CoordinateList[Index].X + 1, ObjectInfo.CoordinateList[Index].Y );
                        ObjectInfo.CoordinateList.push_back( Coordinate );
                        ModelBuffer[( ObjectInfo.CoordinateList[Index].Y * IMAGEWIDTH + ObjectInfo.CoordinateList[Index].X + 1 )] &= 0x00;//šú®øMark
                        this->Boundary( ObjectInfo, Coordinate );
                    }

                    if( ModelBuffer[( ( ObjectInfo.CoordinateList[Index].Y + 1 ) * IMAGEWIDTH + ObjectInfo.CoordinateList[Index].X + 1 )] & ObjectInfo.MaskLabel ){
                        ObjectInfo.CoordinateCnt++;
                        TCoordinate Coordinate( ObjectInfo.CoordinateList[Index].X + 1, ObjectInfo.CoordinateList[Index].Y + 1 );
                        ObjectInfo.CoordinateList.push_back( Coordinate );
                        ModelBuffer[( ( ObjectInfo.CoordinateList[Index].Y + 1 ) * IMAGEWIDTH + ObjectInfo.CoordinateList[Index].X + 1 )] &= 0x00;//šú®øMark
                        this->Boundary( ObjectInfo, Coordinate );
                    }

                    if( ModelBuffer[( ( ObjectInfo.CoordinateList[Index].Y + 1 ) * IMAGEWIDTH + ObjectInfo.CoordinateList[Index].X )] & ObjectInfo.MaskLabel ){
                        ObjectInfo.CoordinateCnt++;
                        TCoordinate Coordinate( ObjectInfo.CoordinateList[Index].X, ObjectInfo.CoordinateList[Index].Y + 1 );
                        ObjectInfo.CoordinateList.push_back( Coordinate );
                        ModelBuffer[( ( ObjectInfo.CoordinateList[Index].Y + 1 ) * IMAGEWIDTH + ObjectInfo.CoordinateList[Index].X )] &= 0x00;//šú®øMark
                        this->Boundary( ObjectInfo, Coordinate );
                    }

                    if( ModelBuffer[( ( ObjectInfo.CoordinateList[Index].Y + 1 ) * IMAGEWIDTH + ObjectInfo.CoordinateList[Index].X - 1 )] & ObjectInfo.MaskLabel ){
                        ObjectInfo.CoordinateCnt++;
                        TCoordinate Coordinate( ObjectInfo.CoordinateList[Index].X - 1, ObjectInfo.CoordinateList[Index].Y + 1 );
                        ObjectInfo.CoordinateList.push_back( Coordinate );
                        ModelBuffer[( ( ObjectInfo.CoordinateList[Index].Y + 1 ) * IMAGEWIDTH + ObjectInfo.CoordinateList[Index].X - 1 )] &= 0x00;//šú®øMark
                        this->Boundary( ObjectInfo, Coordinate );
                    }

                    if( ModelBuffer[( ObjectInfo.CoordinateList[Index].Y * IMAGEWIDTH + ObjectInfo.CoordinateList[Index].X - 1 )] & ObjectInfo.MaskLabel ){
                        ObjectInfo.CoordinateCnt++;
                        TCoordinate Coordinate( ObjectInfo.CoordinateList[Index].X - 1, ObjectInfo.CoordinateList[Index].Y );
                        ObjectInfo.CoordinateList.push_back( Coordinate );
                        ModelBuffer[( ObjectInfo.CoordinateList[Index].Y * IMAGEWIDTH + ObjectInfo.CoordinateList[Index].X - 1 )] &= 0x00;//šú®øMark
                        this->Boundary( ObjectInfo, Coordinate );
                    }
                }
                Index++;
            }
            if( ObjectInfo.Area() > 30 ){
                ObjectInfo.Center.X = ( ObjectInfo.XMax + ObjectInfo.XMin ) / 2;
                ObjectInfo.Center.Y = ( ObjectInfo.YMax + ObjectInfo.YMin ) / 2;
                if( ObjectInfo.MaskLabel == OrangeLabel ){
                    if( !ObjectList[etOrange]->ObjectCnt ){
                        ObjectList[etOrange]->ObjectInfoList.insert( ObjectList[etOrange]->ObjectInfoList.begin(), ObjectInfo );
                    }else{
                        ObjectList[etOrange]->ObjectInfoList.push_back( ObjectInfo );
                    }
                    ObjectList[etOrange]->ObjectCnt++;
                }else if( ObjectInfo.MaskLabel == YellowLabel ){
                    if( !ObjectList[etYellow]->ObjectCnt ){
                        ObjectList[etYellow]->ObjectInfoList.push_back( ObjectInfo );
                    }else{
                        ObjectList[etYellow]->ObjectInfoList.insert( ObjectList[etYellow]->ObjectInfoList.begin(), ObjectInfo );
                    }
                    ObjectList[etYellow]->ObjectCnt++;
                }else if( ObjectInfo.MaskLabel == BlueLabel ){
                    if( !ObjectList[etBlue]->ObjectCnt ){
                        ObjectList[etBlue]->ObjectInfoList.push_back( ObjectInfo );
                    }else{
                        ObjectList[etBlue]->ObjectInfoList.insert( ObjectList[etBlue]->ObjectInfoList.begin(), ObjectInfo );
                    }
                    ObjectList[etBlue]->ObjectCnt++;
                }else if( ObjectInfo.MaskLabel == GreenLabel ){
                    if( !ObjectList[etGreen]->ObjectCnt ){
                        ObjectList[etGreen]->ObjectInfoList.push_back( ObjectInfo );
                    }else{
                        ObjectList[etGreen]->ObjectInfoList.insert( ObjectList[etGreen]->ObjectInfoList.begin(), ObjectInfo );
                    }
                    ObjectList[etGreen]->ObjectCnt++;
                }else if( ObjectInfo.MaskLabel == BlackLabel ){
                    if( !ObjectList[etBlack]->ObjectCnt ){
                        ObjectList[etBlack]->ObjectInfoList.push_back( ObjectInfo );
                    }else{
                        ObjectList[etBlack]->ObjectInfoList.insert( ObjectList[etBlack]->ObjectInfoList.begin(), ObjectInfo );
                    }
                    ObjectList[etBlack]->ObjectCnt++;
                }else if( ObjectInfo.MaskLabel == RedLabel ){
                    if( !ObjectList[etRed]->ObjectCnt ){
                        ObjectList[etRed]->ObjectInfoList.push_back( ObjectInfo );
                    }else{
                        ObjectList[etRed]->ObjectInfoList.insert( ObjectList[etRed]->ObjectInfoList.begin(), ObjectInfo );
                    }
                    ObjectList[etRed]->ObjectCnt++;
                }else if( ObjectInfo.MaskLabel == WhiteLabel ){
                    if( !ObjectList[etWhite]->ObjectCnt ){
                        ObjectList[etWhite]->ObjectInfoList.push_back( ObjectInfo );
                    }else{
                        ObjectList[etWhite]->ObjectInfoList.insert( ObjectList[etWhite]->ObjectInfoList.begin(), ObjectInfo );
                    }
                    ObjectList[etWhite]->ObjectCnt++;
                }else if( ObjectInfo.MaskLabel == OthersLabel ){
                    if( !ObjectList[etOther]->ObjectCnt ){
                        ObjectList[etOther]->ObjectInfoList.push_back( ObjectInfo );
                    }else{
                        ObjectList[etOther]->ObjectInfoList.insert( ObjectList[etOther]->ObjectInfoList.begin(), ObjectInfo );
                    }
                    ObjectList[etOther]->ObjectCnt++;
                }
            }
        }
    }
    bNewFlag = true;
    delete ModelBuffer;
    return true;
}

TObjectInfo::TObjectInfo()
{
    this->Center.X = 0;
    this->Center.Y = 0;
    this->CoordinateCnt = 1;
    this->AreaSize = -1;
}

TObjectInfo::TObjectInfo(int xValue, int yValue)
{
    this->Center.X = 0;
    this->Center.Y = 0;
    TCoordinate Coordinate( xValue, yValue );
    this->CoordinateList.push_back( Coordinate );
    this->CoordinateCnt = 1;
    this->AreaSize = -1;
    this->XMax = xValue;
    this->XMin = xValue;
    this->YMax = yValue;
    this->YMin = yValue;
}


int TObjectInfo::Area()
{
    return this->AreaSize = ( this->ObjectWidth = ( this->XMax - this->XMin ) ) * ( this->ObjectHigh = ( this->YMax - this->YMin ) );
}

TImageInfo::TImageInfo()
{
    this->ObjectCnt = 0;
}
