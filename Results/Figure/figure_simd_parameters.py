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
## PLOT FIGURE 
##############################################################################################################################################################

list_csvfile = ["Inputchannels_Latency_And_Energy.csv", "Inputwidth_Latency_And_Energy.csv", "Kernelsize_Latency_And_Energy.csv", "Filters_Latency_And_Energy.csv", "Groups_Latency_And_Energy.csv"]
list_category = ["Input channels", "Input width", "Kernel size", "Filters", "Groups"]

dico_color = {"Conv" : 'g', "AddConv" : 'b', "GroupConv" : 'c', "DepthwiseConv" : 'r', "ShiftConv" : 'm'}


fig, ax = plt.subplots(nrows=5, ncols=5,figsize=(20, 16))

##############################################################################################################################################################
# Influence of kernel size, input width, input channels and filters,groups (plot for all layer types)
##############################################################################################################################################################

for i in range(len(list_csvfile)):

    dataframe = pd.read_csv(os.getcwd()+os.sep+"Dataframe"+os.sep+list_csvfile[i],sep=",")
    dataframe["Consumption (mJ)"] = dataframe["Consumption (J)"] * 1000 # We multiply by 10e-3 to get mJ
    dataframe["DSP"] = dataframe["DSP"].map(algo_conversion_dict)
    
    category = list_category[i]
    
    ax[0][i].tick_params(axis='y', labelsize=18)
    ax[0][i].tick_params(axis='x', labelsize=18)
    ax[0][i].grid()
    
    ax[1][i].tick_params(axis='y', labelsize=18)
    ax[1][i].tick_params(axis='x', labelsize=18)
    ax[1][i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[1][i].grid()
    
    ax[2][i].tick_params(axis='y', labelsize=18)
    ax[2][i].tick_params(axis='x', labelsize=18)
    ax[2][i].grid()
    
    ax[3][i].tick_params(axis='y', labelsize=18)
    ax[3][i].tick_params(axis='x', labelsize=18)
    ax[3][i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[3][i].grid()
    
    ax[4][i].tick_params(axis='y', labelsize=18)
    ax[4][i].tick_params(axis='x', labelsize=18)
    ax[4][i].set_xlabel(list_category[i], fontsize=16)
    ax[4][i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[4][i].grid()
    
    ax[i][0].set_xticks([4,8,16,32])
    ax[i][1].set_xticks([8,16,24,32])
    ax[i][3].set_xticks([3,7,11])
    ax[i][3].set_xticks([4,8,16,32])
    ax[i][4].set_xticks([4,8,16,32])
    
    if i ==0:
        ax[0][i].set_ylabel('Latency (s)', fontsize=16)
        ax[1][i].set_ylabel('Latency gain', fontsize=16)
        ax[2][i].set_ylabel('Consumption (mJ)', fontsize=16)
        ax[3][i].set_ylabel('Consumption gain', fontsize=16)
        ax[4][i].set_ylabel('Ratio total instructions', fontsize=16)
        
    
    
    if list_category[i]!="Groups":   
        if category!="Input width":
            dataframe["Input width"]=32
        if category!="Input channels":
            dataframe["Input channels"]=16
        if category!="Filters":
            dataframe["Filters"]=16
        if category!="Kernel size":
            dataframe["Kernel size"]=3
        if category!="Groups":
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
        
        #Compute ratio
        a = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Layer type",category])["Total_Instructions"].apply(list)
        ratio_dataframe = a.index.to_frame(index=False)
        a = pd.Series([x if len(x)==3 else [np.nan]+x for x in a])
        ratio_dataframe["Total_Instructions"] = a.tolist()
        ratio_dataframe[['SIMD_Instructions','NAIF_Instructions', 'IM2COL_Instructions']] = pd.DataFrame(ratio_dataframe["Total_Instructions"].tolist(), index=ratio_dataframe.index)
        ratio_dataframe["Ratio_Instructions_im2col_simd"] = ratio_dataframe['IM2COL_Instructions']/ratio_dataframe['SIMD_Instructions']
        
        b = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Layer type",category])["Latency (s)"].apply(list)
        b = pd.Series([x if len(x)==3 else [np.nan]+x for x in b])
        ratio_dataframe["Latency (s)"] = b.tolist()
        ratio_dataframe[['SIMD_Latency','NAIF_Latency', 'IM2COL_Latency']] = pd.DataFrame(ratio_dataframe["Latency (s)"].tolist(), index=ratio_dataframe.index)
        ratio_dataframe["Ratio_Latency_im2col_simd"] = ratio_dataframe['IM2COL_Latency']/ratio_dataframe['SIMD_Latency']
        
        c = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Layer type",category])["Consumption (mJ)"].apply(list)
        c = pd.Series([x if len(x)==3 else [np.nan]+x for x in c])
        ratio_dataframe["Consumption (mJ)"] = c.tolist()
        ratio_dataframe[['SIMD_Consumption','NAIF_Consumption', 'IM2COL_Consumption']] = pd.DataFrame(ratio_dataframe["Consumption (mJ)"].tolist(), index=ratio_dataframe.index)
        ratio_dataframe["Ratio_Consumption_im2col_simd"] = ratio_dataframe['IM2COL_Consumption']/ratio_dataframe['SIMD_Consumption']
        filtered_df = ratio_dataframe[~ratio_dataframe["Ratio_Latency_im2col_simd"].isnull()]
        
        reg1 = LinearRegression().fit(np.array(filtered_df["Ratio_Latency_im2col_simd"]).reshape(-1, 1), np.array(filtered_df["Ratio_Instructions_im2col_simd"]).reshape(-1, 1))
        print(reg1.score(np.array(filtered_df["Ratio_Latency_im2col_simd"]).reshape(-1, 1), np.array(filtered_df["Ratio_Instructions_im2col_simd"]).reshape(-1, 1)))
        
        liste_layer_type = dataframe["Layer type"].sort_values(ascending=True).unique()
        
        for j,layer in enumerate(liste_layer_type):
        
          layer_ratio_dataframe = ratio_dataframe[ratio_dataframe["Layer type"]==layer].sort_values(by=[category], ascending=True)
          
          if layer != "AddConv":
              ax[0][i].plot(layer_ratio_dataframe[category], layer_ratio_dataframe['SIMD_Latency'],'-o', color = dico_color[layer])
              ax[1][i].plot(layer_ratio_dataframe[category], layer_ratio_dataframe["Ratio_Latency_im2col_simd"],'-o', color = dico_color[layer])
              ax[2][i].plot(layer_ratio_dataframe[category], layer_ratio_dataframe['SIMD_Consumption'],'-o', color = dico_color[layer])
              ax[3][i].plot(layer_ratio_dataframe[category], layer_ratio_dataframe["Ratio_Consumption_im2col_simd"],'-o', color = dico_color[layer])
              ax[4][i].plot(layer_ratio_dataframe[category], layer_ratio_dataframe["Ratio_Instructions_im2col_simd"],'-o', color = dico_color[layer])
              
    else:
        dataframe["Input width"]=10
        dataframe["Input channels"]=128
        dataframe["Filters"]=64
        dataframe["Kernel size"]=3
        dataframe["Padding"]=1
        dataframe["Channelsbygroup"]=dataframe["Input channels"]/dataframe["Groups"]
        dataframe["Filtersbygroup"]=dataframe["Filters"]/dataframe["Groups"]
        dataframe["r"]=dataframe["Padding"]/dataframe["Kernel size"]
        dataframe["s"]=1-dataframe["r"]
        dataframe["t"]=dataframe["Kernel size"]-dataframe["Padding"]
        dataframe["x"]=dataframe["Input width"]-2*dataframe["Padding"]
        dataframe["y"]=dataframe["x"]**2/dataframe["Input width"]**2
        dataframe["z"]=1-dataframe["y"]
        
        instruction_tot = [layer_dict["GroupConv"][row["DSP"]]["TOTAL"](row["Input width"],row["Input channels"],row["Filters"],row["Kernel size"],row["Groups"],row["Padding"],row["Channelsbygroup"],row["Filtersbygroup"],row["r"],row["s"],row["t"],row["x"],row["y"],row["z"]) for i, row in dataframe.iterrows()]
        dataframe["Total_Instructions"] = instruction_tot
        load_instruction = [layer_dict["GroupConv"][row["DSP"]]["LOAD"](row["Input width"],row["Input channels"],row["Filters"],row["Kernel size"],row["Groups"],row["Padding"],row["Channelsbygroup"],row["Filtersbygroup"],row["r"],row["s"],row["t"],row["x"],row["y"],row["z"]) for i, row in dataframe.iterrows()]
        dataframe["Load_Instructions"] = load_instruction
        store_instruction = [layer_dict["GroupConv"][row["DSP"]]["STR"](row["Input width"],row["Input channels"],row["Filters"],row["Kernel size"],row["Groups"],row["Padding"],row["Channelsbygroup"],row["Filtersbygroup"],row["r"],row["s"],row["t"],row["x"],row["y"],row["z"]) for i, row in dataframe.iterrows()]
        dataframe["Store_Instructions"] = store_instruction
        mac_instruction = [layer_dict["GroupConv"][row["DSP"]]["MACS"](row["Input width"],row["Input channels"],row["Filters"],row["Kernel size"],row["Groups"],row["Padding"],row["Channelsbygroup"],row["Filtersbygroup"],row["r"],row["s"],row["t"],row["x"],row["y"],row["z"]) for i, row in dataframe.iterrows()]
        branch_instruction = [layer_dict["GroupConv"][row["DSP"]]["BRANCH"](row["Input width"],row["Input channels"],row["Filters"],row["Kernel size"],row["Groups"],row["Padding"],row["Channelsbygroup"],row["Filtersbygroup"],row["r"],row["s"],row["t"],row["x"],row["y"],row["z"]) for i, row in dataframe.iterrows()]
        dataframe["Branch_Instructions"] = branch_instruction
        dataframe["MAC"] = mac_instruction
        dataframe["Load_Ratio"] = dataframe["Load_Instructions"]/dataframe["Total_Instructions"]
        dataframe["Store_Ratio"] = dataframe["Store_Instructions"]/dataframe["Total_Instructions"]
        dataframe["Branch_Ratio"] = dataframe["Branch_Instructions"]/dataframe["Total_Instructions"]
        dataframe["MAC_Ratio"] = dataframe["MAC"]/dataframe["Total_Instructions"]
        
        #Compute ratio
        a = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Groups"])["Total_Instructions"].apply(list)
        ratio_dataframe = a.index.to_frame(index=False)
        ratio_dataframe["Total_Instructions"] = a.tolist()
        ratio_dataframe[['SIMD_Instructions','NAIF_Instructions', 'IM2COL_Instructions']] = pd.DataFrame(ratio_dataframe["Total_Instructions"].tolist(), index=ratio_dataframe.index)
        ratio_dataframe["Ratio_Instructions_im2col_simd"] = ratio_dataframe['IM2COL_Instructions']/ratio_dataframe['SIMD_Instructions']
        
        b = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Groups"])["Latency (s)"].apply(list)
        ratio_dataframe["Latency (s)"] = b.tolist()
        ratio_dataframe[['SIMD_Latency','NAIF_Latency', 'IM2COL_Latency']] = pd.DataFrame(ratio_dataframe["Latency (s)"].tolist(), index=ratio_dataframe.index)
        ratio_dataframe["Ratio_Latency_im2col_simd"] = ratio_dataframe['IM2COL_Latency']/ratio_dataframe['SIMD_Latency']
        
        c = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Groups"])["Consumption (mJ)"].apply(list)
        c = pd.Series([x if len(x)==3 else [np.nan]+x for x in c])
        ratio_dataframe["Consumption (mJ)"] = c.tolist()
        ratio_dataframe[['SIMD_Consumption','NAIF_Consumption', 'IM2COL_Consumption']] = pd.DataFrame(ratio_dataframe["Consumption (mJ)"].tolist(), index=ratio_dataframe.index)
        ratio_dataframe["Ratio_Consumption_im2col_simd"] = ratio_dataframe['IM2COL_Consumption']/ratio_dataframe['SIMD_Consumption']
        
        #Plot
        ax[0][i].plot(ratio_dataframe[category], ratio_dataframe['SIMD_Latency'],'-o', color = 'c')
        ax[1][i].plot(ratio_dataframe[category], ratio_dataframe["Ratio_Latency_im2col_simd"],'-o', color = 'c')
        ax[2][i].plot(ratio_dataframe[category], ratio_dataframe['SIMD_Consumption'],'-o', color = 'c')
        ax[3][i].plot(ratio_dataframe[category], ratio_dataframe["Ratio_Consumption_im2col_simd"],'-o', color = 'c')
        ax[4][i].plot(ratio_dataframe[category], ratio_dataframe["Ratio_Instructions_im2col_simd"],'-o', color = 'c')
        
      
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

fig.legend(handles=[Conv, AddConv, DepthwiseConv, GroupConv, ShiftConv],loc="lower center",ncol=5, fontsize= "large",bbox_to_anchor=(0.45, 0.01))

fig.text(0.17, 0.9, "a)",fontsize = 28)
fig.text(0.34, 0.9, "b)",fontsize = 28)
fig.text(0.51, 0.9, "c)",fontsize = 28)
fig.text(0.66, 0.9, "d)",fontsize = 28)
fig.text(0.83, 0.9, "e)",fontsize = 28)

fig.text(0.05, 0.81, "1)",fontsize = 28)
fig.text(0.05, 0.65, "2)",fontsize = 28)
fig.text(0.05, 0.50, "3)",fontsize = 28)
fig.text(0.05, 0.33, "4)",fontsize = 28)
fig.text(0.05, 0.16, "5)",fontsize = 28)

fig.subplots_adjust(wspace=0.3,hspace=0.45)


fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+'simdparameter.jpeg',bbox_inches='tight',pad_inches = 0, format='jpeg')
fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+'simdparameter.eps',bbox_inches='tight',pad_inches = 0, format='eps')