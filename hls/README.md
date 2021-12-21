# HLS

## Model data
* [Pretrained parameters for HLS](https://drive.google.com/file/d/1texDvaILZbqv1XFNordmTOCJ4tLplYpg/view?usp=sharing)
* [Model info](https://docs.google.com/spreadsheets/d/1eIjE4K5dIcSPoDpqoz2H_D20m_ik5UpiCrTiqRX3LSs/edit#gid=0)

## Sources

* run.h, run.cpp : Top module
* buffer.h : Defines the BRAM array length and data type
* model_param.h : Defines scale factors for the `bb_conv_weight` (see Model Info)
* model_shape.h : Defines the shape of the model (e.g., kernel size, stride, and padding)
* basicblock.h, basicblock.cpp : Implementation of the basicblock
