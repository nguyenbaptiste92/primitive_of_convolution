import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter

from sklearn.linear_model import LinearRegression

##############################################################################################################################################################
## DICTIONNARY OF INSTRUCTIONS
# The formula comes from the files in the Assembly_Instructions_Count folder
# a = dim_im_out
# b = ch_im_in
# c = ch_im_out
# d = dim_kernel
# g = groups
# p = padding
# h = b/g (channel by group)
# i = c/g (filter by group)
# r = p/d
# s = 1 - r
# t = d-p
# x = a-2p
# y = (a-2p)**2/a**2
# z = 1-y
##############################################################################################################################################################

algo_conversion_dict = {
   0 : "NAIF",
   1 : "IM2COL",
   2 : "SIMD",
}

##############################################################################################################################################################
## CONVOLUTION
##############################################################################################################################################################

naive_conv_dict = {
    "BRANCH" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(d*((y*d+z*t)*(2*b+1)+2)+4)+1)+3)),
    "STR" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(d*((y*d+z*t)*(4*b+5)+4)+6)+4)+1)+19),
    "LOAD" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(d*((y*d+z*t)*(7*b+11)+7)+23)+7)+4)+15),
    "MULT" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(d*((y*d+z*t)*b+2)+6)+2))+4),
    "SMLAD": (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 0),
    "TOTAL" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(d*((y*d+z*t)*(16*b+24)+21)+71)+21)+14+49)),
    "MACS" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a**2*b*c*d**2),
}

im2col_conv_dict = {
    "BRANCH" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(x*(d*(b*d+4)+1/2*(b*c*d**2+c+3)+2)+1)+4*p*(a*(d*(d*(r*(2*b+2)+s*(b+3)+3)+2)+1/2*(b*c*d**2+c+1)+2)+3)+2),
    "STR" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(x*(b*d**2+1/2*(5/2*b*c*d**2+5*c+11)+3))+4*p*(a*(d*(d*(r*b+s*b)+1)+1/2*(5/2*b*c*d**2+5*c+11)+2)+6)+18),
    "LOAD" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(x*(d*(b*d+5)+1/2*(5*b*c*d**2+8*c+11)+2)+9)+4*p*(a*(d*(d*(s*(b+2)+5)+3)+1/2*(5*b*c*d**2+8*c+15)+4)+12)+18),
    "MULT" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(x*(2*d+1/2*(2*b*c*d**2)+1)+1)+4*p*(a*(d*(d*s+1)+1/2*(2*b*c*d**2)+1)+1)+2),
    "SMLAD": (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 0),
    "TOTAL" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(x*(d*(4*b*d+20)+1/2*(23/2*b*c*d**2+30*c+47)+7)+23)+4*p*(a*(d*(d*(r*(4*b+7)+s*(5*b+13)+14)+10)+1/2*(23/2*b*c*d**2+30*c+53)+13)+34)+55),
    "MACS" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a**2*b*c*d**2),
}

simd_conv_dict = {
    "BRANCH" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(x*(d*(1/2*b*d+4)+1/2*(1/4*b*c*d**2+3/2*c+3)+2)+1)+4*p*(a*(d*(d*(r*(2*b+2)+s*(1/2*b+3)+3)+12)+1/2*(1/4*b*c*d**2+3/2*c+1)+2)+3)+2),
    "STR" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(x*(1/4*b*d**2+1/2*(1/2*b*c*d**2+13/2*c+17))+3)+4*p*(a*(d*(d*(r*b+1/4*s*b)+1)+1/2*(1/2*b*c*d**2+13/2*c+17)+2)+6)+18),
    "LOAD" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(x*(d*(1/4*b*d+5)+1/2*(13/8*b*c*d**2+13*c+14)+2)+9)+4*p*(a*(d*(d*(s*(1/4*b+2)+5)+3)+1/2*(13/8*b*c*d**2+13*c+18)+4)+12)+18),
    "MULT" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(x*(2*d+1)+1)+4*p*(a*(d*(d*s+1)+2)+1)+2),
    "SMLAD": (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 1/2*a**2*b*c*d**2),
    "TOTAL" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(x*(d*(9/4*b*d+25)+1/2*(9/2*b*c*d**2+61/2*c+60)+7)+21)+4*p*(a*(d*(d*(r*(4*b+7)+s*(9/4*b+19)+14)+10)+1/2*(9/2*b*c*d**2+61/2*c+66)+13)+34)+55),
    "MACS" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 1/2*a**2*b*c*d**2),
}

conv_dict = {
    "NAIF" : naive_conv_dict,
    "IM2COL" : im2col_conv_dict,
    "SIMD" : simd_conv_dict,
}

##############################################################################################################################################################
## GROUPCONVOLUTION
##############################################################################################################################################################

naive_groupconv_dict = {
    "BRANCH" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(d*((y*d+z*t)*(2*h+1)+2)+4)+2)+2)),
    "STR" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(d*((y*d+z*t)*(4*h+5)+5)+8)+8)+5)+23),
    "LOAD" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(d*((y*d+z*t)*(7*h+8)+11)+19)+18)+7)+22),
    "MULT" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(d*((y*d+z*t)*h)+5)+3)+1)+7),
    "SMLAD": (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 0),
    "TOTAL" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(d*((y*d+z*t)*(16*h+21)+24)+62)+46)+23)+64),
    "MACS" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a**2*h*c*d**2),
}

im2col_groupconv_dict = {
    "BRANCH" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d*(h+4)+2)+1/2*(1/2*i*(2*d**2*h+2)+3)+2)+3)+4*p*(a*(d*(d*(r*(2*h+2)+s*(h+3)+6)+2)+1/2*(1/2*i*(2*d**2*h+2)+1)+2)+3)+5)+4),
    "STR" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d*h+1)+1/2*(1/2*i*(4*d**2*h+10)+12)+2)+4)+4*p*(a*(d*(d*(r*h+s*h)+1)+1/2*(1/2*i*(4*d**2*h+10)+12)+2)+4)+10)+20),
    "LOAD" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d*(h+4)+4)+1/2*(1/2*i*(9*d**2*h+16)+16)+5)+11)+4*p*(a*(d*(d*(s*h+3)+4)+1/2*(1/2*i*(9*d**2*h+16)+16)+5)+8)+20)+27),
    "MULT" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d+1)+d**2*h*i+1)+2)+4*p*(a*(d+d**2*h*i+1)+2)+6)+9),
    "SMLAD": (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 0),
    "TOTAL" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d*(4*h+22)+11)+1/2*(1/2*i*(20*d**2*h+67)+59)+14)+33)+4*p*(a*(d*(d*(r*(4*h+7)+s*(4*h+15)+18)+11)+1/2*(1/2*i*(20*d**2*h+67)+55)+15)+27)+63)+81),
    "MACS" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a**2*h*c*d**2),
}

simd_groupconv_dict = {
    "BRANCH" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d*(1/2*h+4)+2)+1/2*(1/4*h*i*d**2+i+3)+2)+2)+4*p*(a*(d*(d*(r*(2*h+2)+s*(1/2*h+3)+6)+2)+1/2*(1/4*h*i*d**2+i+1)+2)+3)+1)+4),
    "STR" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d*(1/4*h+1))+1/2*(5/8*h*i*d**2+7/2*i+16)+2)+7)+4*p*(a*(d*(d*(r*(h+1)+s*(1/4*h+1))+1)+1/2*(5/8*h*i*d**2+7/2*i+16)+2)+4)+5)+21),
    "LOAD" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d*(1/4*h+4)+4)+1/2*(7/4*h*i*d**2+8*i+19)+5)+19)+4*p*(a*(d*(d*(s*(1/4*h+3)+4)+4)+1/2*(7/4*h*i*d**2+8*i+19)+5)+11)+9)+27),
    "MULT" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d+1)+1)+3)+4*p*(a*(d*(d*s+1)+1)+1)+5)+9),
    "SMLAD": (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 1/2*a**2*h*c*d**2),
    "TOTAL" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d*(9/4*h+29)+10)+1/2*(19/4*h*i*d**2+22*i+69)+13)+46)+4*p*(a*(d*(d*(r*(4*h+8)+s*(9/4*h+21)+18)+11)+1/2*(19/4*h*i*d**2+22*i+65)+14)+30)+32)+83),
    "MACS" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 1/2*a**2*h*c*d**2),
}

groupconv_dict = {
    "NAIF" : naive_groupconv_dict,
    "IM2COL" : im2col_groupconv_dict,
    "SIMD" : simd_groupconv_dict,
}

##############################################################################################################################################################
## ADDCONVOLUTION
##############################################################################################################################################################

naive_addconv_dict = {
    "BRANCH" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(d*(d*(2*b*y+y+b*z+2*z+5)+2)+4)+2)+2)),
    "STR" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(d*(d*(3*b*y+y+b*z+3)+3)+7)+7)+7)+23),
    "LOAD" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(d*(d*(6*b*y+2*b*z+3*z+6)+10)+17)+15)+12)+20),
    "MULT" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 3*a**2*c+9),
    "SMLAD": (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 0),
    "TOTAL" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(d*(d*(19*b*y+5*y+11*b*z+7*z+21)+21)+57)+37)+30)+64),
    "MACS" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a**2*b*c*d**2),
}

im2col_addconv_dict = {
    "BRANCH" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(x*(1/2*(b*c*d**2+c+3)+d*(b*d+4)+2)+1)+4*p*(a*(1/2*(b*c*d**2+c+1)+d*(d*(2*r*b+s*b+2*r+3*s+3)+2)+2)+3)+2),
    "STR" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(x*(1/2*(2*b*c*d**2+5*c+12)+b*d**2)+1)+4*p*(a*(1/2*(2*b*c*d**2+5*c+12)+d*(d*(r*b+s*b)+1)+2)+6)+18),
    "LOAD" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(x*(1/2*(11/2*b*c*d**2+8*c+12)+d*(b*d+5)+1)+4)+4*p*(a*(1/2*(11/2*b*c*d**2+8*c+16)+d*(d*(s*b+2*s+5)+3)+4)+12)+18),
    "MULT" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(2*d*x+1)+4*p*(a*(d*(d*s+1)+1)+1)+2),
    "SMLAD": (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 0),
    "TOTAL" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(x*(1/2*(16*b*c*d**2+33*c+49)+d*(4*b*d+20)+6)+13)+4*p*(a*(1/2*(16*b*c*d**2+33*c+55)+d*(d*(4*r*b+5*s*b+7*r+13*s+14)+10)+13)+34)+55),
    "MACS" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a**2*b*c*d**2),
}

addconv_dict = {
    "NAIF" : naive_addconv_dict,
    "IM2COL" : im2col_addconv_dict,
}

##############################################################################################################################################################
## DEPTHWISECONVOLUTION
##############################################################################################################################################################

naive_depthwiseconv_dict = {
    "BRANCH" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(b*(d*(2*(d*y+t*z)+2)+5)+2)+2) + c*(a*(a*(1*(1*(2*b+1)+2)+4)+1)+3)),
    "STR" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(b*(d*(3*(d*y+t*z)+2)+10)+5)+5)+19 + c*(a*(a*(1*(1*(4*b+5)+4)+6)+4)+1)+19),
    "LOAD" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(b*(d*(9*(d*y+t*z)+8)+24)+9)+13)+14 + c*(a*(a*(1*(1*(7*b+11)+7)+23)+7)+4)+15),
    "MULT" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(b*(d*((d*y+t*z)+2)+4)+1)+1)+5 + c*(a*(a*(1*(1*b+2)+6)+2))+4),
    "SMLAD": (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 0),
    "TOTAL" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(b*(d*(22*(d*y+t*z)+22)+81)+27)+28)+45 + c*(a*(a*(1*(1*(16*b+24)+21)+71)+21)+14+49)),
    "MACS" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a**2*b*d**2 + a**2*b*c),
}

im2col_depthwiseconv_dict = {
    "BRANCH" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(d*(d*(b*y+3*y+2*b*z+2*z+6)+2)+1/2*(b*d**2+2*b)+2)+3)+2 + a*(a*(1*(b+4)+1/2*(b*c+c+3)+2)+1)+2),
    "STR" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(d*(d*(b*y+b*z)+1)+1/2*(2*b+6)+1)+5)+16 + a*(a*(b+1/2*(5/2*b*c+5*c+11)+3))+18),
    "LOAD" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(d*(d*(2*b*y+2)+3)+1/2*(3*b*d**2+8*b+4)+6)+9)+15 +  a*(a*(1*(b*1+5)+1/2*(5*b*c+8*c+11)+2)+9)+18),
    "MULT" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(d*(d*y+1)+1/2*(2*b*d**2+1)+1)+1)+3 + a*(a*(2+1/2*(2*b*c)+1)+1)+2),
    "SMLAD": (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 0),
    "TOTAL" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(d*(d*(4*b*y+16*y+4*b*z+8*z+17)+10)+1/2*(9*b*d**2+36*c+20)+15)+27)+49 + a*(a*(1*(4*b+20)+1/2*(23/2*b*c+30*c+47)+7)+23)+55),
    "MACS" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a**2*b*d**2 + a**2*b*c),
}

simd_depthwiseconv_dict = {
    "BRANCH" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(d*(d*(b*y+3*y+2*b*z+2*z+6)+2)+1/4*b*(d**2+2)+1)+3)+2 + a*(a*(1/2*(1/4*b*c+3/2*c+1)+1/2*b+4)+3)+1),
    "STR" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(d*(d*(b*y+b*z))+1/4*b*(5/2*d**2+10)+5)+6)+19 + a*(a*(1/2*(1/2*b*c+13/2*c+17)+1/4*b+4)+1)+4),
    "LOAD" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(d*(d*(3*y+z+2)+1)+1/4*b*(3/2*d**2+8)+5)+2)+2 + a*(a*(1/2*(13/8*b*c+13*c+13)+19/4*b+7)+7)),
    "MULT" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(d*(d*y+1)+1)+1)+1),
    "SMLAD": (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 1/2*a**2*b*d**2 + 1/2*a**2*b*c),
    "TOTAL" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(d*(d*(4*b*y+16*y+4*b*z+8*z+17)+10)+1/4*b*(24*d**2+49)+32)+28)+53 + a*(a*(1/2*(9/2*b*c+61/2*c+56)+9/4*b+33)+9)+25),
    "MACS" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 1/2*a**2*b*d**2 + 1/2*a**2*b*c),
}

im2col_conv_dict = {
    "BRANCH" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(x*(d*(b*d+4)+1/2*(b*c*d**2+c+3)+2)+1)+4*p*(a*(d*(d*(r*(2*b+2)+s*(b+3)+3)+2)+1/2*(b*c*d**2+c+1)+2)+3)+2),
    "STR" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(x*(b*d**2+1/2*(5/2*b*c*d**2+5*c+11)+3))+4*p*(a*(d*(d*(r*b+s*b)+1)+1/2*(5/2*b*c*d**2+5*c+11)+2)+6)+18),
    "LOAD" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(x*(d*(b*d+5)+1/2*(5*b*c*d**2+8*c+11)+2)+9)+4*p*(a*(d*(d*(s*(b+2)+5)+3)+1/2*(5*b*c*d**2+8*c+15)+4)+12)+18),
    "MULT" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(x*(2*d+1/2*(2*b*c*d**2)+1)+1)+4*p*(a*(d*(d*s+1)+1/2*(2*b*c*d**2)+1)+1)+2),
    "SMLAD": (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 0),
    "TOTAL" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : x*(x*(d*(4*b*d+20)+1/2*(23/2*b*c*d**2+30*c+47)+7)+23)+4*p*(a*(d*(d*(r*(4*b+7)+s*(5*b+13)+14)+10)+1/2*(23/2*b*c*d**2+30*c+53)+13)+34)+55),
    "MACS" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a**2*b*c*d**2),
}

depthwiseconv_dict = {
    "NAIF" : naive_depthwiseconv_dict,
    "IM2COL" : im2col_depthwiseconv_dict,
    "SIMD" : simd_depthwiseconv_dict,
}

##############################################################################################################################################################
## SHIFTCONVOLUTION
##############################################################################################################################################################

naive_shiftconv_dict = {
    "BRANCH" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(b+3)+3)+2)),
    "STR" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(b+1)+1)+5)+9),
    "LOAD" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(12*b+5)+5)+6)+6),
    "MULT" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 3*a**2*b*c+2),
    "SMLAD": (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 0),
    "TOTAL" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(26*b+22)+14)+22)+28),
    "MACS" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a**2*b*c),
}

im2col_shiftconv_dict = {
    "BRANCH" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(3*b+5+1/2*(b*c+c+1))+3)+1),
    "STR" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(2*b+1/2*(5/2*b*c+5*c+11))+2)+11),
    "LOAD" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(9*b+3+1/2*(5*b*c+17/2*c+14))+2)+8),
    "MULT" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(2*b+1/2*(2*b*c+1)))+1),
    "SMLAD": (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 0),
    "TOTAL" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(29*b+23+1/2*(23/2*b*c+65/2*c+48))+13)+39),
    "MACS" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a**2*b*c),
}

simd_shiftconv_dict = {
    "BRANCH" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(1/2*(1/4*b*c+3/2*c+1)+11/4*b+3)+3)+1),
    "STR" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(1/2*(1/2*b*c+13/2*c+17)+9/4*b+5)+1)+4),
    "LOAD" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(1/2*(13/8*b*c+13*c+13)+19/4*b+11)+7)),
    "MULT" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a**2*(9/4*b+1)),
    "SMLAD": (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 1/2*a**2*b*c),
    "TOTAL" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a*(a*(1/2*(9/2*b*c+61/2*c+56)+109/4*b+34)+9)+25),
    "MACS" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 1/2*a**2*b*c),
}

shiftconv_dict = {
    "NAIF" : naive_shiftconv_dict,
    "IM2COL" : im2col_shiftconv_dict,
    "SIMD" : simd_shiftconv_dict,
}

layer_dict = {
    "Conv" : conv_dict,
    "AddConv" : addconv_dict,
    "DepthwiseConv" : depthwiseconv_dict,
    "GroupConv" : groupconv_dict,
    "ShiftConv" : shiftconv_dict,
}

##############################################################################################################################################################
## TEST
##############################################################################################################################################################

"""a = 32
b = 16
c = 16
d = 3
p = 1
g = 1
h = b/g
i = c/g
r = p/d
s = 1-r
t = d-p
x = a-2*p
y = (a-2*p)**2/a**2
z = 1-y

for key in shiftconv_dict.keys():
  for instructions in shiftconv_dict[key].keys():
      print(key,instructions)
      print(shiftconv_dict[key][instructions](a,b,c,d,g,p,h,i,r,s,t,x,y,z))"""
      
##############################################################################################################################################################
## PLOT FIGURE FOR GROUP CONVOLUTIONS
##############################################################################################################################################################
   
#csvfile = "Inputwidth_Latency_And_Energy.csv"
#save_file = "instructions_input_width_im2col"
#category = "Input width"
   
csvfile = "Inputchannels_Latency_And_Energy.csv"
save_file = "instructions_input_channels_im2col"
category = "Input channels"

#csvfile = "Kernelsize_Latency_And_Energy.csv"
#save_file = "instructions_kernel_size_im2col"
#category = "Kernel size"

#csvfile = "Filters_Latency_And_Energy.csv"
#save_file = "instructions_filters_im2col"
#category = "Filters"

dico_color = {"Conv" : 'g', "AddConv" : 'b', "GroupConv" : 'c', "DepthwiseConv" : 'r', "ShiftConv" : 'm'}

dataframe = pd.read_csv(os.getcwd()+os.sep+"Dataframe"+os.sep+csvfile,sep=",")
dataframe["Consumption (mJ)"] = dataframe["Consumption (J)"] * 1000 # We multiply by 10e-3 to get mJ
dataframe["DSP"] = dataframe["DSP"].map(algo_conversion_dict)


###COMPLETE DATAFRAME
if category!="Input width":
    dataframe["Input width"]=32
if category!="Input channels":
    dataframe["Input channels"]=16
if category!="Filters":
    dataframe["Filters"]=16
if category!="Kernel size":
    dataframe["Kernel size"]=3
dataframe["Groups"]=2
dataframe["Channelsbygroup"]=dataframe["Input channels"]/dataframe["Groups"]
dataframe["Filtersbygroup"]=dataframe["Filters"]/dataframe["Groups"]
dataframe["Padding"]=np.floor(dataframe["Kernel size"]/2)
dataframe["r"]=dataframe["Padding"]/dataframe["Kernel size"]
dataframe["s"]=1-dataframe["r"]
dataframe["t"]=dataframe["Kernel size"]-dataframe["Padding"]
dataframe["x"]=dataframe["Input width"]-2*dataframe["Padding"]
dataframe["y"]=dataframe["x"]**2/dataframe["Input width"]**2
dataframe["z"]=1-dataframe["y"]

### ADD INSTRUCTIONS
instruction_tot = [layer_dict[row["Layer type"]][row["DSP"]]["TOTAL"](row["Input width"],row["Input channels"],row["Filters"],row["Kernel size"],row["Groups"],row["Padding"],row["Channelsbygroup"],row["Filtersbygroup"],row["r"],row["s"],row["t"],row["x"],row["y"],row["z"]) for i, row in dataframe.iterrows()]
dataframe["Total_Instructions"] = instruction_tot
load_instruction = [layer_dict[row["Layer type"]][row["DSP"]]["LOAD"](row["Input width"],row["Input channels"],row["Filters"],row["Kernel size"],row["Groups"],row["Padding"],row["Channelsbygroup"],row["Filtersbygroup"],row["r"],row["s"],row["t"],row["x"],row["y"],row["z"]) for i, row in dataframe.iterrows()]
dataframe["Load_Instructions"] = load_instruction
store_instruction = [layer_dict[row["Layer type"]][row["DSP"]]["STR"](row["Input width"],row["Input channels"],row["Filters"],row["Kernel size"],row["Groups"],row["Padding"],row["Channelsbygroup"],row["Filtersbygroup"],row["r"],row["s"],row["t"],row["x"],row["y"],row["z"]) for i, row in dataframe.iterrows()]
dataframe["Store_Instructions"] = store_instruction 
mac_instruction = [layer_dict[row["Layer type"]][row["DSP"]]["MACS"](row["Input width"],row["Input channels"],row["Filters"],row["Kernel size"],row["Groups"],row["Padding"],row["Channelsbygroup"],row["Filtersbygroup"],row["r"],row["s"],row["t"],row["x"],row["y"],row["z"]) for i, row in dataframe.iterrows()]
dataframe["MAC"] = mac_instruction
branch_instruction = [layer_dict[row["Layer type"]][row["DSP"]]["BRANCH"](row["Input width"],row["Input channels"],row["Filters"],row["Kernel size"],row["Groups"],row["Padding"],row["Channelsbygroup"],row["Filtersbygroup"],row["r"],row["s"],row["t"],row["x"],row["y"],row["z"]) for i, row in dataframe.iterrows()]
dataframe["Branch_Instructions"] = branch_instruction
dataframe["Load_Ratio"] = dataframe["Load_Instructions"]/dataframe["Total_Instructions"]
dataframe["Store_Ratio"] = dataframe["Store_Instructions"]/dataframe["Total_Instructions"]
dataframe["Branch_Ratio"] = dataframe["Branch_Instructions"]/dataframe["Total_Instructions"]
dataframe["MAC_Ratio"] = dataframe["MAC"]/dataframe["Total_Instructions"]
#print(dataframe.sort_values(by=["Layer type",'DSP',category], ascending=False))

print(dataframe.sort_values(by=["Layer type",'DSP',category], ascending=False)[["Layer type","DSP", category, "Load_Ratio","Store_Ratio","Branch_Ratio","MAC_Ratio"]])

#dataframe[['DSP', 'Layer type', 'Kernel size', 'Latency (s)', 'Total_Instructions', 'Load_Ratio', 'Store_Ratio', 'Branch_Ratio','MAC_Ratio']].to_csv("dataframe.csv")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
#Compute ratio
a = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Layer type",category])["Total_Instructions"].apply(list)
ratio_dataframe = a.index.to_frame(index=False)
a = pd.Series([x if len(x)==3 else [np.nan]+x for x in a])
ratio_dataframe["Total_Instructions"] = a.tolist()
ratio_dataframe[['SIMD_Instructions','NAIF_Instructions', 'IM2COL_Instructions']] = pd.DataFrame(ratio_dataframe["Total_Instructions"].tolist(), index=ratio_dataframe.index)
ratio_dataframe["Ratio_Instructions_im2col_simd"] = ratio_dataframe['IM2COL_Instructions']/ratio_dataframe['SIMD_Instructions']
ratio_dataframe["Ratio_Instructions_naif_im2col"] = ratio_dataframe['NAIF_Instructions']/ratio_dataframe['IM2COL_Instructions']

b = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Layer type",category])["Latency (s)"].apply(list)
b = pd.Series([x if len(x)==3 else [np.nan]+x for x in b])
ratio_dataframe["Latency (s)"] = b.tolist()
ratio_dataframe[['SIMD_Latency','NAIF_Latency', 'IM2COL_Latency']] = pd.DataFrame(ratio_dataframe["Latency (s)"].tolist(), index=ratio_dataframe.index)
ratio_dataframe["Ratio_Latency_im2col_simd"] = ratio_dataframe['IM2COL_Latency']/ratio_dataframe['SIMD_Latency']

c = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Layer type",category])["Load_Instructions"].apply(list)
c = pd.Series([x if len(x)==3 else [np.nan]+x for x in c])
ratio_dataframe["Load_Instructions"] = c.tolist()
ratio_dataframe[['SIMD_Load','NAIF_Load', 'IM2COL_Load']] = pd.DataFrame(ratio_dataframe["Load_Instructions"].tolist(), index=ratio_dataframe.index)
ratio_dataframe["Ratio_Load_im2col_simd"] = ratio_dataframe['IM2COL_Load']/ratio_dataframe['SIMD_Load']
ratio_dataframe["Ratio_Load_naif_im2col"] = ratio_dataframe['NAIF_Load']/ratio_dataframe['IM2COL_Load']

d = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Layer type",category])["Store_Instructions"].apply(list)
d = pd.Series([x if len(x)==3 else [np.nan]+x for x in d])
ratio_dataframe["Store_Instructions"] = d.tolist()
ratio_dataframe[['SIMD_Store','NAIF_Store', 'IM2COL_Store']] = pd.DataFrame(ratio_dataframe["Store_Instructions"].tolist(), index=ratio_dataframe.index)
ratio_dataframe["Ratio_Store_im2col_simd"] = ratio_dataframe['IM2COL_Store']/ratio_dataframe['SIMD_Store']

e = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Layer type",category])["Branch_Instructions"].apply(list)
e = pd.Series([x if len(x)==3 else [np.nan]+x for x in e])
ratio_dataframe["Branch_Instructions"] = e.tolist()
ratio_dataframe[['SIMD_Branch','NAIF_Branch', 'IM2COL_Branch']] = pd.DataFrame(ratio_dataframe["Branch_Instructions"].tolist(), index=ratio_dataframe.index)
ratio_dataframe["Ratio_Branch_im2col_simd"] = ratio_dataframe['IM2COL_Branch']/ratio_dataframe['SIMD_Branch']

f = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Layer type",category])["Load_Ratio"].apply(list)
ratio_dataframe["Load_Ratio"] = f.tolist()
ratio_dataframe[['SIMD_Load_Ratio','NAIF_Load_Ratio', 'IM2COL_Load_Ratio']] = pd.DataFrame(ratio_dataframe["Load_Ratio"].tolist(), index=ratio_dataframe.index)

g = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Layer type",category])["Branch_Ratio"].apply(list)
ratio_dataframe["Branch_Ratio"] = g.tolist()
ratio_dataframe[['SIMD_Branch_Ratio','NAIF_Branch_Ratio', 'IM2COL_Branch_Ratio']] = pd.DataFrame(ratio_dataframe["Branch_Ratio"].tolist(), index=ratio_dataframe.index)

h = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Layer type",category])["MAC_Ratio"].apply(list)
ratio_dataframe["MAC_Ratio"] = h.tolist()
ratio_dataframe[['SIMD_MAC_Ratio','NAIF_MAC_Ratio', 'IM2COL_MAC_Ratio']] = pd.DataFrame(ratio_dataframe["MAC_Ratio"].tolist(), index=ratio_dataframe.index)

print(ratio_dataframe[["Layer type",category,"Ratio_Latency_im2col_simd","Ratio_Instructions_im2col_simd","Ratio_Load_im2col_simd","Ratio_Store_im2col_simd","Ratio_Branch_im2col_simd"]])

#reg = LinearRegression().fit(ratio_dataframe["Ratio_Instructions_naif_im2col"].values.reshape(-1,1), ratio_dataframe["Ratio_Latency_naif_im2col"].values.reshape(-1,1))
#print(reg.score(ratio_dataframe["Ratio_Instructions_naif_im2col"].values.reshape(-1,1), ratio_dataframe["Ratio_Latency_naif_im2col"].values.reshape(-1,1)))
#reg = LinearRegression().fit(ratio_dataframe["Ratio_Load_naif_im2col"].values.reshape(-1,1), ratio_dataframe["Ratio_Latency_naif_im2col"].values.reshape(-1,1))
#print(reg.score(ratio_dataframe["Ratio_Load_naif_im2col"].values.reshape(-1,1), ratio_dataframe["Ratio_Latency_naif_im2col"].values.reshape(-1,1)))

fig, ax = plt.subplots(nrows=1, ncols=4,figsize=(20, 4))

ax[2].tick_params(axis='y', labelsize=18)
ax[2].tick_params(axis='x', labelsize=18)
ax[2].set_ylabel('Total instructions ratio', fontsize=14)
ax[2].set_xlabel(category, fontsize=20)
ax[2].grid()

ax[3].tick_params(axis='y', labelsize=18)
ax[3].tick_params(axis='x', labelsize=18)
ax[3].set_ylabel('Load instructions ratio', fontsize=14)
ax[3].set_xlabel(category, fontsize=20)
ax[3].grid()

ax[0].tick_params(axis='y', labelsize=18)
ax[0].tick_params(axis='x', labelsize=18)
ax[0].set_ylabel('Naive MAC instructions (%)', fontsize=14)
ax[0].set_ylim([0.0, 0.2])
ax[0].set_xlabel(category, fontsize=20)
ax[0].grid()

ax[1].tick_params(axis='y', labelsize=18)
ax[1].tick_params(axis='x', labelsize=18)
ax[1].set_ylabel('Im2Col MAC instructions (%)', fontsize=14)
ax[1].set_ylim([0.0, 0.2])
ax[1].set_xlabel(category, fontsize=20)
ax[1].grid()

liste_layer_type = ratio_dataframe["Layer type"].sort_values(ascending=True).unique()
# Plot
for j,layer in enumerate(liste_layer_type):

  layer_ratio_dataframe = ratio_dataframe[ratio_dataframe["Layer type"]==layer].sort_values(by=[category], ascending=True)
  layer_ratio_dataframe.dropna()
  
  if layer != "AddConv":
      ax[0].plot(layer_ratio_dataframe[category], layer_ratio_dataframe['NAIF_MAC_Ratio'],'-o', color = dico_color[layer])
      ax[1].plot(layer_ratio_dataframe[category], layer_ratio_dataframe['IM2COL_MAC_Ratio'],'-o', color = dico_color[layer])
  
  ax[2].plot(layer_ratio_dataframe[category], layer_ratio_dataframe["Ratio_Instructions_naif_im2col"],'-o', color = dico_color[layer])
  ax[3].plot(layer_ratio_dataframe[category], layer_ratio_dataframe["Ratio_Load_naif_im2col"],'-o', color = dico_color[layer])
  
GroupConv = mlines.Line2D([], [], color='c', marker='s', linestyle='None',
                          markersize=12, label='Grouped convolution')

Conv = mlines.Line2D([], [], color='g', marker='s', linestyle='None',
                          markersize=12, label='Convolution')
                          
DepthwiseConv = mlines.Line2D([], [], color='r', marker='s', linestyle='None',
                          markersize=12, label='Depthwise separable convonvolution')
                          
ShiftConv = mlines.Line2D([], [], color='m', marker='s', linestyle='None',
                          markersize=12, label='Shift convolution')
                          
AddConv = mlines.Line2D([], [], color='b', marker='s', linestyle='None',
                          markersize=12, label='Add convolution')

fig.legend(handles=[Conv, AddConv, DepthwiseConv, GroupConv, ShiftConv],loc="lower center",ncol=5, fontsize= "x-large",bbox_to_anchor=(0.48, 0.00))

fig.subplots_adjust(wspace=0.35,bottom=0.3)

fig.text(0.20, 0.9, "a)",fontsize = 32)
fig.text(0.40, 0.9, "b)",fontsize = 32)
fig.text(0.60, 0.9, "c)",fontsize = 32)
fig.text(0.80, 0.9, "d)",fontsize = 32)


fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+save_file+'.jpeg',bbox_inches='tight',pad_inches = 0, format='jpeg')
fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+save_file+'.eps',bbox_inches='tight',pad_inches = 0, format='eps')
