#include "nnom.h"

#define GROUP_CONV2D_KERNEL_0 {39, 72, 73, -5, 55, 11, -72, 35, -64, 54, 70, 50, -34, 23, -26, -3, -28, -61, 46, -32, -10, -36, -22, -40, 47, -17, 53, 55, 38, 29, 8, 65, -14, 38, -30, -67, 39, -5, -69, -8, 5, 21, -35, 6, 52, -4, -58, -48, 18, -28, 51, 12, -6, -44, 24, -66, -64, 37, -57, 55, 48, -40, 2, -70, 27, 34, 72, -50, 70, 0, 63, 46, 63, 55, -17, -1, 13, -40, -39, 37, -15, -36, -41, -58, 70, 52, -35, -46, 12, 64, 63, -35, 31, 28, 25, 11, 35, -20, -20, -5, -13, -50, 15, -26, 18, 14, -7, -7, -51, 11, -56, 36, 57, 49, -29, 67, 38, -21, -45, 35, 19, -45, 11, 0, 23, 41, 31, 45, 15, -1, -10, -12, -65, -25, 61, -52, -18, -45, 39, 73, 63, 24, 52, -27, 21, 19, 30, 37, 49, -22, -21, 49, -64, -72, -14, -66, 67, 36, -53, 36, 35, 70, -22, 34, 7, 12, -1, -5, -47, 26, 25, 42, -54, 4, -35, -14, 38, -71, 43, -7, 34, -59, 58, 2, 52, 1, 70, 70, -42, 34, -62, -44, -28, 32, 54, -5, -63, 21, 37, 20, -41, -29, -23, 46, 12, 7, -64, 37, 52, -73, -63, -9, 19, -40, -63, -49, 8, -73, 31, 33, 4, -10, 36, 2, 41, -20, -32, -21, 49, 24, 29, -10, -21, 31, -26, -20, 22, 35, 29, 71, 33, 26, -32, -23, 52, -9, 66, 4, -10, 4, 7, -33, 17, -37, -12, -34, -9, 57, -68, 16, 58, 35, -66, 45, 38, 31, -1, -57, -15, -41, 62, 42, 21, -46, 15, -46, 36, -55, 54, 48, 51, 29, 14, 21, 20, -38, -69, 69, 64, 37, -44, 20, -23, 29, -73, -3, 71, 22, 7, -16, 7, 73, 34, -13, -67, 63, 55, -10, -48, -50, 70, -40, -64, 56, -19, -15, -21, 14, 50, -56, 11, 60, 9, 31, 14, -42, -51, -33, -5, 37, 60, -5, -31, 18, 56, 34, 43, -7, 5, 22, 59, -71, 20, -45, -53, 42, 34, 63, -64, 8, 18, 27, 36, -4, -38, -38, -40, 16, -25, 62, -14, 58, -38, 61, 0, -73, -16, 0, 13, 37, -70, 42, 37, -47, 10, 40, -4, 25, 51, -15, 48, 33, -40, -56, 18, -37, -14, -26, 4, 32, 60, 45, 51, -22, 38, 40, 70, -30, -27, -67, 37, -55, 18, -14, -39, 38, -69, 13, 21, 23, -10, 34, -50, -24, 1, -19, -39, -5, 29, -2, -69, 63, 35, 16, 43, -21, -9, 37, -1, 22, 2, 55, 46, 34, -38, -61, 65, -71, -40, 24, -22, 39, -71, 21, 44, -27, 73, 46, -22, -22, -59, 0, 3, 66, 45, 62, -36, 50, -47, 9, -10, 25, -12, -19, 47, 53, 13, -38, -73, -12, -21, -16, 12, 7, 27, -69, -38, -29, -56, 52, 19, 47, -55, 44, -45, -46, 72, 26, -17, -65, -70, 36, 49, 62, 15, -3, 52, 55, 24, 35, -51, 72, -41, -62, -50, -38, 61, 14, 24, -7, -57, -41, -56, -3, -6, 5, 2, 21, 54, 15, -67, 68, -12, 60, 65, -47, 1, 3, -19, 34, -9, -5, 23, -26, 21, 33, -14, 58, 68, -26, 3, 61, 60, -61, -69, -20, -9, 64, 50, 63, -1, -52, 34, 23, 53, -2, -40, 51, -32, 5, -31, -14, -7, 23, 45, -58, -3, -40, -69, 70, -57, 19, 63, 66, 32, 14, -71, -29, -26, -53, -58, 27, 66, -8, 42, -12, -23, -57, -41, 24, -62, 66, 49, 38, -4, 29, -24, 27, 69, 16, 71, -20, 13, 18, -62, 4, -41, 11, -60, 4, 15, 0, -50, -45, 20, 10, 20, 32, 43, -22, 0, -21, -10, -23, 45, 2, 54, 71, 41, -25, 31, -43, 52, -46, -55, -20, 62, -49, 35, -5, -44, -10, 51, -6, -57, 39, -25, 64, 19, -23, 41, 32, 41, 66, 65, 42, -69, -4, 33, 13, -40, 3, 6, 56, -25, 34, -47, 44, -52, 17, -8, 49, -44, 14, -26, 37, 63, 1, 7, 31, -7, -43, -63, 34, 62, 40, 42, -58, -34, 28, 72, -21, 4, 68, -39, 9, -18, 4, 57, 19, 46, -47, 6, -18, 6, -11, 42, 35, -35, 19, -62, -4, 41, -19, -48, -27, -6, -47, -24, -51, -69, 60, -37, -11, -35, 4, -45, 44, 26, 53, -11, 60, -48, -40, -32, -32, -20, -47, 64, 55, -42, 36, -9, -66, -1, 1, -2, 54, 55, 17, 55, -33, 70, -47, 6, -64, -48, -20, -69, 27, 23, 10, 23, 46, -44, 8, 6, -64, -11, -22, 46, 39, -30, -18, -6, -23, -68, -58, 16, 12, -13, 8, -24, -70, -61, -3, -58, -13, 38, -7, 52, 45, 45, -1, -11, 19, 59, 23, -36, -14, -62, -5, -45, -32, -67, -33, -22, 29, -6, 11, 31, -41, -11, 23, -3, 18, -69, 53, -49, -28, 54, 67, 40, 42, 50, 5, -52, -44, -32, -56, -27, -65, 52, -57, -21, -32, 1, -44, 11, -66, -49, 8, -3, 33, 53, 16, -30, -61, 1, 14, 18, -72, -58, -65, -31, -3, 48, -24, -21, 1, 22, -45, 69, -53, -71, 52, 25, -25, 24, 1, 36, 33, -1, -26, 16, 28, 45, -1, 70, 8, 5, -6, -13, -8, 3, 13, 45, 15, -15, 42, 29, 40, -17, -63, -25, -16, -19, -27, -47, -38, 11, 42, -20, -34, -10, 34, -9, 30, -55, 72, 30, -20, -55, 21, 50, -44, -69, 25, -45, -57, -8, 13, 18, 21, 67, 1, -70, -57, 35, 29, -13, -72, -59, 29, -24, -33, 67, -22, 33, 15, -11, 26, -47, -5, 36, -57, -57, 17, -22, -1, 65, 60, -73, -14, 3, -23, 17, 25, 31, -3, -25, -68, -50, -47, -60, 5, -56, 25, 53, 14, 65, 8, 33, -48, -18, 55, 53, -9, 68, -58, -29, -39, 26, 5, 23, 53, -52, -20, -58, -1, 32, 37, -61, 61, 70, -9, -42, -69, 14, -63, -4, -33, 19, -24, -61, -66, 54, 4, 62, -72, 35, -43, -37, -14, -16, 32, -6, 15, 45, -14, 44, -54, -17, -35, 45, 13, -36, 30, -12, -51, -68, 33, -24, 13, 36, -52, 39, -12, -16, -5, -30, 62, -24, -73, -31, 20, 4, -40, -68, 40, -7, 28, 15, 49, 38, 13, 17, 52, -8, -21, 20, -50, -44, 41, 29, 25, -31, 55, 37, 17, 1, 10, 29, -46, 0, -55, 8, 24, 53, 68, 28, 22, 41, -14, 53, -54, -58, -27, 60, -32, 24, 14, 26, -17, -51, -47, 71, 60, -42, 24, 64, -49, 31, -26, -43, -1, -71, 16, 41, -13, 59, -34, -64, -46, 51, 60, -13, -60, 39, 22, -31, 50, 40, 8, -16, 54, -49, 42, 11, -49, 68, 44, -31, -42, -67, 7, 53, 19, 45, -53, -39, -70, -6, -40, 54, 10, -19, -50, 64, 35, -39, -55, 19, 14, 1, 2, -27, -43, -27, -63, 17, 14, -72, -66, 56, -69, 49, -43, -63, 72, 20, 4, 61, 45, 52, -26, 24, -25, -72, 33, -20, -41, -46, -60, -15, -66, -4, -40, 7, -41, 10, -72, 7, -43, 37, -7, -32, 57, -53, 39, -10, -68, -23, 42, -73, -66, 49, -61, 36, 67, 37, 43, -59, -6, 58, 43, 20, -12, 46, -64, -71, -23, 26, 71, 35, 49, 49, -58, -23, 67, -33, -54, -2, 12, 31, -8, 9, -55, -48, -15, 42, -59, 25, 68, 37, -56, -64, -23, -65, 21, 44, -62, 9, 54, 63, -37, -34, 65, 69, 41, 43, -45, -17, -52, 50, -49, -12, -20, -14, -46, -4, -29, -45, -40, 49, -41, -21, 54, 37, 10, 47, -14, 30, -9, 11, -26, 66, -65, 42, 68, -58, -62, 33, -25, 13, 8, 51, 1, 19, 31, -19, 10, 32, -36, -2, 10, -31, -35, 42, 69, 37, -9, -13, 43, -31, -44, 28, 31, 2, -34, -4, 43, -56, 11, 23, 5, -5, -13, 59, 59, 72, 32, -73, 1, 42, -24, -7, 13, -48, 2, 55, -67, -8, -7, -31, -26, -68, -42, 73, 12, 27, -62, -59, 30, -24, 69, 56, 18, -46, -37, 40, -11, 7, 3, 17, -7, 0, 10, 65, 71, 41, -9, -43, 48, -24, 56, 3, 18, 46, -14, -8, -58, 53, -48, -65, 45, -15, -9, 13, -54, 19, 49, -31, 71, -45, -32, 48, 52, 28, 49, -17, 66, -2, 26, 60, -58, 6, -54, 48, -6, 12, 23, -73, -7, -6, 4, -19, -66, 8, -17, -24, -46, 19, 35, 29, -63, 65, 52, -35, 67, 46, -42, 24, 69, -16, 41, -4, -26, -33, -40, -10, 59, -32, 26, -36, -32, 56, 65, 54, -38, 25, 32, -9, 60, 67, -31, 41, 21, 63, -2, -7, -66, 55, -14, -40, -3, 73, 31, -35, -56, 59, 70, -42, 24, -27, 70, -33, 36, -4, -31, -35, -12, 48, 10, 1, -19, 39, -38, 8, -43, 27, 3, -48, 43, 20, 15, -58, 72, -12, 43, -55, -63, -50, -3, -37, 31, -29, -23, -55, -28, 61, 38, -61, -21, 31, -52, -28, 50, 39, -50, -60, 55, -11, 3, -63, -65, -47, 52, 65, -52, -61, -3, -34, -59, -21, 21, 47, -46, 52, -72, 45, -11, -44, 18, 51, -26, -10, 15, 50, 38, 37, -14, 25, 67, 31, 4, 35, -49, -44, 26, -42, -13, -13, 66, 35, 22, -44, 37, -3, 49, -8, 18, 31, 59, 62, -56, 17, -18, 48, 52, 4, -28, 66, 16, -29, -18, -65, -18, 71, -28, 32, 60, 18, -2, 52, -24, -11, -38, 36, -63, -61, -55, 30, 60, 69, 73, -37, 13, 6, -7, 8, -15, -15, 42, -2, -13, 61, 10, 26, 34, -27, 31, -10, -49, 31, 31, -68, -6, 9, 65, 60, 13, 46, 55, 58, 16, 54, -28, -26, -36, -29, -63, -46, 51, 22, 19, -50, -18, -36, 57, 54, -19, -22, -20, 42, -59, -47, 17, 31, -56, -23, -10, -67, -39, -46, 69, 48, 3, 8, 66, 11, 54, -62, -64, -15, -62, 58, 62, -7, 57, 29, 65, -68, -56, 72, 58, -12, 0, 8, -33, -19, -24, -2, -20, -13, -61, 44, -27, -10, 43, -23, 70, 59, 19, -1, -7, 29, 66, -18, -40, 48, -6, -71, -67, 16, 59, -17, 7, -6, -59, 47, 34, 7, -12, -71, 16, -60, 28, -43, -19, -26, -70, -54, -14, -67, -4, -5, 29, -41, -3, -21, -61, 55, 43, -62, -60, 0, 2, -29, 22, 32, -30, 1, 71, 20, 66, 51, 29, 48, -40, 60, 11, 38, -9, -12, -2, -60, -73, 71, -20, -33, -43, -1, 60, 71, 68, -9, 51, 54, -33, -12, 1, -20, -68, 51, -26, 23, -12, 66, -7, 7, -9, 71, 40, -47, -60, 30, -14, 73, -43, -10, -4, -50, -12, -54, 48, -25, 38, 1, -72, 47, -16, -61, -1, -31, -55, -41, 73, 54, 35, -67, -62, 37, -9, -2, 21, -46, 41, -27, 18, -39, 8, -73, -51, -10, 6, 3, 29, -1, -2, -70, -27, -28, -40, -21, -8, 11, 72, -13, -13, -56, 31, 4, 20, 0, 64, -34, -6, 30, -52, -22, -21, -36, -71, 11, 58, -18, -69, -12, -20, 28, -39, -5, 68, 17, -34, -19, 34, 47, -55, -52, -26, 63, -3, 42, -52, 8, 55, 71, 29, 42, 69, -35, -44, 16, -45, 21, 9, -36, 11, -16, -18, -2, 50, 45, -4, 14, 63, -7, 43, -69, -66, -16, -13, 35, -55, -40, 2, 35, -4, 17, 64, -37, 65, 36, -49, -7, -73, 45, 26, 1, 14, 48, -62, 39, 59, 52, 50, 33, 31, -11, -66, -12, -14, -10, -68, 20, -66, -57, 61, 4, -59, -45, -40, -45, 13, 22, -24, 25, -23, 52, -48, -71, -21, 64, 59, 6, 61, -20, 48, -72, -3, 12, 13, 41, -16, 1, -41, -23, 62, -61, -36, -52, -37, -49, 13, -54, -34, -64, 64, -20, 28, -15, 58, 65, -21, -8, -45, 34, -32, 21, -53, -7, -42, 22, 20, -68, 13, 4, -29, 40, 56, 47, 51, 56, 6, 54, -51, -62, 69, 7, -50, 13, 50, -31, -50, -35, 69, -57, 10, 22, 20, 16, 31, -39, -30, -17, -44, 4, 65, 58, -48, 71, 29, 70, 51, -62, -35, 24, -56, 33, -7, -35, -37, -27, 14, -24, -20, 73, 15, -44, 48, -40, -18, 67, 8, -37, -13, 57, 16, 44, -51, 27, -55, -69, -49, -12, 4, 38, -12, 25, 57, -3, -41, -2, 61, 0, -23, -50, -62, 14, 16, 70, 71, 3, 4, 51, -28, 6, 53, 40, -54, -65, -24, -33, 68, 41, 27, -47, 36, 6, 65, 48, 33, 59, 60, 44, 20, -32, 24, 71, 1, -5, 46, 59, -29, 34, -39, -31, -59, 18, 62, 71, 67, -69, 17, 37, -66, -26, 45, 33, -3, -41, -39, 27, 23, -24, -7, -27, 58, -41, -20, 8, 36, 18, 39, -46, -31, 52, -10, -48, -10, 50, -10, 27, -20, 29, -40, -31, -46, -59, -7, 66, -68, -61, 71, -52, -28, -13, 6, 47, -69, -59, 1, -8, -54, -68, 30, 27, 7, 7, -68, -47, -65, 33, -39, 45, -68, -7, 5, -42, -64, 31, 40, 71, -5, 44, 19, 67, -7, 2, 20, -27, 49, -31, 45, -61, 67, -45, 52, -58, -53, -62, 59, -62, 43, -31, -36, 67, -69, 4, 60, -15, 23, -14, 9, 15, -17, -35, 13, -42, -20, -55, 7, 65, -48, -72, -66, -53, -53, -19, 73, 68, 38, -16, 61, -19, 6, -14, -56, -1, -33, -11, -65, -25, 38, 49, 45, 33, 72, 39, -20, -36, 18, 10, -26, -73, 37, 0, -48, 50, -45, 49, 27, -72, -64, -73, -29, -58, 45, 48, 39, 7, -53, -27, -24, -22, -62, 71, -40, -2, 70, 53, -1, 43, -12, 4, -45, -30, -47, 16, -52, 1, -5, -17, -70, -42, -26, 11, -15, -38, 61, -17, 26, 62, -23, -15, 1, 6, 23, 28, -68, -7, -15, 37, 30, -13, 57, 56, -19, -65, -22, 47, 55, -4, 24, 4, -17, -67, -42, -26, 19, -65, 29, 4, 12, -67, 66, 15, 18, 53, 0, -19, 52, 48, -61, -61, 53, -51, 69, -64, 61, 10, 37, 73, -62, 48, 35, 28, -35, 54, -69, -49, 21, 58, 3, -40, 66, 54, 57, -41, -23, 46, 25, 12, 1, 63, -31, 53, 43, 20, 22, 63, -44, -69, 29, -13, 37, 34, -13, -3, 23, 20, -40, 13, -35, 27, -55, 36, -24, 16, 40, -11, 7, 34, 53, -14, 70, -43, -17, 6, 33, 42, -59, 33, -1, -68, 11, 9, 52, 31, 53, 3, -26, -14, 9, -68, -46, -27, 64, 9, -3, -11, -38, 38, -72, 55, -54, 36, 42, 39, 41, -34, -55, -31, -55, 4, -31, -29, -47, -24, -42, -23, 33, -6, -47, -30, 39, 41, -21, 64, -44, -67, -13, 38, -40, -33, -66, -35, -38, 28, 37, -33, -49, -5, -51, 27, 66, 50, 19, -67, 38, 11, 46, -59, -3, 46, -59, 64, -63, -37, -46, -6, -14, 35, -57, 4, -51, 50, -11, 46, 28, -30, -8, -14, -28, -9, -20, 37, 56, -59, -13, -2, -24, 13, 13, -5, -62, -64, 21, -28, 45, 14, -58, 23, -23, 67, -42, -57, 24, 35, -14, 0, 17, 10, -73, 55, 70, -51, -9, 5, 33, -71, 61, 18, -40, -38, -1, 72, 21, 32, -27, 23, -25, 63, 63, -9, -31, 32, -31, 57, -56, 24, 41, 1, -59, 60, -63, -67, -48, 73, -21, 35, -19, -1, 2, -47, 21, -66, -43, 46, -34, -32, 72, 17, -8, -36, 3, -32, 3, -17, 1, -72, 24, 49, -53, -63, -5, -12, 46, -33, -44, 23, -58, -29, 47, 36, 38, 40, -23, 3, 39, 29, -10, -41, 61, 27, -10, -13, -9, -35, 7, 64, -34, 13, 17, 46, 59, -11, 72, 31, 13, -55, -5, 11, -11, 33, 38, -22, -43, -42, -55, 45, 16, 31, 0, -42, -63, -50, 48, 16, 0, -65, -11, -52, 32, -31, -38, 27, -67, 71, -66, 59, 64, -42, 51, 4, -22, 38, -45, -32, 5, -58, -55, 59, -70, -72, -44, -30, -11, -26, -9, -52, 51, -45, -42, 17, 1, -25, 33, 28, -23, -65, -41, 14, 68, 67, -49, -34, -35, -25, -19, 45, -23, 1, -28, 12, -28, -17, -60, 7, 46, 5, -58, 22, 64, -22, 27, 5, 72, -49, -56, -33, 66, 24, -44, 24, -18, -69, -22, -34, -3, 8, 59, 53, -65, -13, 36, -46, -55, 43, -46, 70, -37, -29, 72, -55, -33, 22, 39, 69, -51, 65, 18, -53, 65, 10, -1, -14, -55, -59, -10, 72, 46, -57, -65, -35, 72, -72, -2, 60, -24, 26, 30, 26, 6, -45, -14, 43, 57, -60, -48, -44, 68, 12, -13, 28, 35, -30, 24, 38, -21, -59, -17, 73, -31, 3, -49, 13, 66, 44, 68, 44, 43, -68, 52, -50, 35, 68, 49, 41, 44, -65, -52, -16, 72, -10, 58, -63, 58, 35, 51, 11, 30, 43, 35, 56, 29, 62, -12, -60, 52, 21, 6, 61, -9, -42, 56, 21, 38, 44, 35, 0, -48, 52, 4, -29, 16, 65, 23, -11, 57, -46, 6, 26, 57, -59, -6, -11, -7, -48, 31, -9, 30, 73, 44, 62, 47, 73, -3, -27, 69, 67, 61, 29, -17, -50, -20, -71, -58, 40, -34, -43, -4, -16, -53, 23, 30, 56, 19, 6, 55, 71, 56, 56, -26, -56, 5, -34, 40, 2, -38, -13, 72, -34, 30, 61, 23, -10, 33, 22, -5, -45, 18, 1, 30, -68, 37, 8, -44, -61, -11, -67, 22, 69, 31, -4, 17, 67, 54, -13, -40, -21, -29, 17, 65, 5, 34, 47, -72, -4, -54, 20, -48, -68, -72, -43, 43, 31, 66, 57, 63, -37, 15, -62, -15, -67, -34, -62, 12, -16, -6, -42, -16, -73, 70, 27, 35, -51, 69, -32, 14, 16, 0, 68, -35, 69, 21, 62, -53, 65, -51, 48, -63, -23, -17, 37, -40, 50, -52, 41, 5, 73, 68, 21, 10, 60, -50, -8, 68, 52, -44, -61, 20, 56, -31, -34, 43, 10, 17, 59, -1, -35, 66, -52, -50, 7, 42, 49, 6, -47, -20, -19, 51, -48, -48, 30, -40, -28, 16, 37, -53, -66, -15, 6, -70, 15, -21, 4, 52, -21, -45, -55, 12, 45, -30, -56, 43, -32, -12, -28, 53, 48, 60, 5, -57, -67, -24, 30, 8, -54, 36, -7, 38, -45, 30, -45, -28, -45, -44, 13, 27, 25, 45, 31, 60, 53, -2, 47, -62, 14, 20, -63, 72, -6, -54, 69, 53, -25, -9, 56, -69, -5, 48, -60, 61, -9, -27, -35, -68, -29, 20, -28, -61, 53, -40, 68, 66, 4, -62, 36, -71, 22, 39, 52, -41, -28, 67, -12, -27, -10, -16, -67, -53, 10, 73, 2, -37, -47, 49, -23, 2, -31, -73, 53, 25, -50, 70, 25, -26, -1, 21, 47, 1, 44, 55, 10, 13, 25, -33, -37, -1, -55, 63, -9, -39, -20, 51, 41, 60, 2, 14, -65, -60, 11, 22, 17, 43, 52, 57, -18, 32, -25, 13, -13, 61, 72, -7, -31, 59, -24, -62, 2, -21, -8, 17, 15, -7, -58, 67, -6, 45, 73, 41, 35, -8, -11, 59, -40, 58, -52, -41, 12, -21, 65, 30, 4, -58, -11, 70, -7, -35, -29, -68, 65, 31, -62, -49, 0, -51, -63, -7, -69, -65, 5, -70, -11, -6, 31, 21, -41, 61, -3, -29, 55, -39, 52, 1, -27, 49, 29, -38, -35, -27, -20, -69, -28, 3, 43, -6, 43, 54, 41, -57, 69, -35, -17, 16, -11, -56, -58, -16, 4, 64, -42, -73, 14, 64, 49, 26, -66, -40, -49, 52, -59, -32, 16, 37, 5, 48, -50, 5, -38, -44, -13, -14, 3, -25, 28, -11, 25, 55, -17, 61, 19, 45, 40, 11, 1, -67, -25, -26, 3, -23, -15, -33, -11, -59, 44, 50, 60, -41, 45, -53, 68, 0, -18, -11, 58, -4, -12, -68, 45, -14, 3, -66, -42, 26, -35, -61, 68, 10, 3, -57, -26, 20, 36, -68, -61, 2, 55, 35, 9, 0, -20, 55, 68, -48, -40, 64, -38, 22, 70, -23, 53, -16, 73, -22, -8, 71, 43, -35, -34, -71, -31, -56, -53, 26, 59, 62, 48, -8, -26, -39, -8, 50, 18, 30, 45, 71, 0, 63, -30, -69, 0, 52, -5, -30, 8, -27, -16, 23, 27, 5, 53, 2, 8, 54, -57, 67, -22, 67, -66, -14, -28, -38, -12, -45, 5, 61, 53, -42, 22, -69, 47, -45, -13, -64, 29, -68, -42, 49, -35, -52, -50, -52, 46, 26, -50, 62, -22, 36, 59, -23, 55, 13, -27, -63, 48, 72, 60, 10, 52, 40, 50, 8, 35, 20, -43, -51, 23, 31, -17, 66, 14, 1, -16, -2, -7, 12, 27, -47, 12, -44, 55, -73, -42, -48, -15, -57, -11, 5, 10, 8, -59, -31, 46, 64, -58, -40, -63, -47, -24, -43, 19, -63, 1, -44, -40, -40, -24, -32, 73, -32, -25, -39, 21, -8, 4, -42, -46, 71, 57, 56, -38, -62, -62, -1, -35, -1, -16, 71, -12, 6, 23, 14, 59, -60, 26, -61, 48, -47, -8, 51, -54, 51, 6, 30, 70, -68, 7, -59, -17, 30, -58, 63, 46, 2, -8, 47, -73, 19, -32, -55, 67, -14, 70, -7, -61, 18, -48, -11, 30, -2, -9, 27, -11, 49, 24, 18, 59, 52, 2, 36, 73, -71, 2, 50, -63, 72, 2, -14, 40, -43, 26, 51, -62, -12, -53, -64, -14, 6, 33, 15, -43, 45, -57, 4, -44, -2, 48, -44, 20, -68, -40, 30, -4, -40, 31, -58, 27, -42, 13, -21, -47, 2, 70, -60, -59, -10, -17, -44, 46, -28, 0, 28, -8, 6, -57, -32, -50, 62, -16, -55, 4, 73, 69, 6, 34, 45, 29, 42, -4, 30, -37, 71, -36, 7, 45, -22, -46, 5, -40, 29, -46, 64, 41, -25, 64, -17, -41, 39, 56, -71, -2, -2, -47, -34, -71, -47, -65, -48, -63, -47, -34, -54, -25, 50, -15, -53, 67, -55, 20, 56, -71, 0, 2, 27, -42, 23, 6, -62, 57, 62, -56, -33, -59, -40, -44, 57, 38, -39, -43, -30, -58, 1, -45, -29, -36, 46, 39, 72, 21, 29, 10, 65, 54, 6, -52, 33, -52, -37, 67, -69, 47, -21, -31, -22, -49, 3, -29, -9, 28, 9, 2, -6, -24, -64, 65, -59, 10, -47, -2, 36, -50, -44, 65, 11, -1, -16, 59, -65, -68, -13, -64, 15, -17, -25, 21, -69, -13, 33, 5, 21, -2, -10, -22, 63, -63, -32, 57, 53, 23, 10, 1, -35, -67, 30, 41, -48, 16, -17, -69, 45, 0, -62, 9, -14, -21, 54, -66, 33, -40, 32, -62, 32, 2, 29, -63, 1, 66, -27, -20, 0, -65, 14, -19, 49, 64, -29, 45, 8, 11, -8, -48, 44, 24, -60, -33, -45, -41, 30, -61, 45, 16, -50, 0, 71, 7, 43, -7, -60, -41, 30, 61, -38, -5, -63, -67, 63, -27, -59, 56, -54, -42, 68, -70, -62, -48, 20, 25, 49, -26, -9, -20, 1, 52, -45, 13, 50, -67, -38, 30, 59, 18, -16, 11, 3, -22, -21, -65, 43, 12, 57, 35, 26, -2, 57, 69, -57, 4, 31, 20, 69, -32, 25, -72, -56, 9, -66, 29, -22, -5, 34, -36, 3, -3, 39, -14, 5, -48, 20, -30, -31, 18, -2, -29, -3, 3, 64, -27, -12, -44, -15, -73, -59, 68, -18, 2, -46, -66, -50, 21, 28, 1, 0, 2, 22, 43, 28, 18, 22, -21, -46, 22, -17, 51, 72, 61, -4, -26, -42, -33, -23, 35, 50, 30, -65, -55, -65, 37, -72, 66, 3, -61, -47, -41, 38, -5, -13, 71, 2, -6, 45, -17, -46, -20, 69, -47, 35, 64, 21, -33, 3, 19, 71, 62, 54, -52, 8, 41, -35, -33, 46, 41, 67, -61, 27, -72, -26, 6, -70, -42, 66, -43, -48, 5, 63, -37, -54, 43, 56, -36, 17, -9, -22, 28, -60, 70, 30, -21, 33, 66, -45, 67, 42, -47, -23, -25, -56, -2, -31, -14, 43, 27, -46, -35, 16, -16, 46, -54, 59, -61, 52, -45, -48, 51, -37, -2, -66, 29, -52, 47, 8, -18, 31, 64, -64, 2, -39, 67, 22, 57, 61, 42, 38, -8, 1, -56, 66, -31, 21, -57, 31, -49, 11, -12, 24, 69, -70, 67, 1, 4, 42, 59, -39, -17, -42, 62, 15, 3, -56, -37, -7, 46, -53, 73, -11, 50, 56, -29, 33, -48, -36, 17, 45, 8, -16, 46, -13, 44, -33, -24, -11, -23, 51, 4, -65, 55, 21, -60, 58, -22, 62, 41, -32, 23, -19, -29, -69, 20, 56, 4, -45, -4, -32, 17, -63, -27, -59, 69, 61, -6, 34, 58, 24, 35, -18, -67, -60, 14, 13, -57, -12, -21, 15, -22, 53, -26, -23, 3, 18, -55, 41, 24, -20, 53, 13, -7, 25, 33, -37, -56, -42, -21, -45, -2, -21, 23, 71, -22, 14, 23, 42, 16, 40, -38, 37, 72, 26, -69, -29, 42, 48, -22, -3, 5, -18, 45, -2, 64, -13, -38, 72, 14, -55, 56, -67, -69, -17, -16, -58, -46, -37, 23, 21, 8, -46, 31, 52, 15, 68, 58, 30, -35, -21, -54, -25, 0, -28, 35, -16, -59, 51, -36, -14, -2, -51, 17, -14, 7, 12, 20, 70, -41, -7, -59, 57, 26, -2, 35, 21, -31, -41, -1, 42, 70, -4, -5, -12, 20, -57, 55, 40, 24, 45, 3, 71, 15, -43, 52, -64, -18, -23, -34, -30, -61, -31, 41, 70, 50, 40, -26, -5, -24, -7, -42, 48, 1, -55, -42, -19, 57, 54, -69, -27, 29, -22, -25, 30, -60, -22, -40, 23, 52, -3, 25, 6, 12, 60, 37, 41, -54, 21, 18, 2, -4, 40, 51, 54, -45, -50, 51, 18, -52, -73, 11, 10, -46, 20, -14, -37, 23, 53, 4, -14, 20, -39, -44, 46, -26, -41, -35, 71, -35, 28, -41, 61, 39, -35, -63, -14, 72, -23, 54, 46, 6, -7, 30, 30, 29, 64, 63, 58, 72, 20, -67, -2, -65, 45, -62, 13, 23, -35, -6, -71, 56, -20, 72, -22, 19, -49, -47, -36, 54, 45, -65, -5, 11, -64, 32, -51, -49, 69, -30, -57, 73, 54, 59, -25, -39, 67, -46, 41, 36, -66, -2, -6, 72, -17, -5, 56, 71, -62, -19, 14, 54, -38, -26, 71, -41, -56, 4, -52, 29, -54, -72, 66, -57, -59, 15, -45, 38, -16, -67, 56, 43, 28, 6, 48, -49, -52, -7, -25, -69, 43, -67, -43, -67, 50, 43, 40, -66, 51, 65, -49, 37, 57, -62, -64, 40, 31, -51, 19, -50, -53, 40, -14, 20, -62, -44, 65, -66, 5, 27, 52, -72, -62, 71, -32, 20, -61, -59, 23, 7, 16, -1, -11, -7, 46, -49, -51, 11, -4, -23, -2, 50, -54, -22, -19, -6, 73, 64, 52, 14, 53, -41, 5, 29, 7, -11, 13, 43, -17, 49, 63, 13, -42, -66, 56, -69, 65, -46, -40, 26, 6, -62, 18, -67, 22, -37, -71, -67, 45, -32, 71, -27, -58, 49, 46, -22, 38, 2, -9, 17, -41, -70, -5, -37, -22, -34, -39, 33, 31, -21, -67, 4, 35, 2, -25, 63, 2, -68, 30, -66, -23, 6, -70, 41, -61, -57, -49, -61, 44, -25, -3, -18, -7, 72, -25, -66, 48, -11, 16, -51, 0, 38, 16, -32, -57, 0, 49, 68, 25, -68, 4, 38, -11, 59, -57, -32, 27, -47, 6, -72, -37, 56, -54, -63, -7, -40, -24, -36, 54, 26, -10, -5, -42, 16, 42, -65, -18, -56, -34, 12, -28, 35, 31, -1, 1, 11, 38, 11, 21, 20, -43, 32, -50, 15, 41, -60, -53, 54, 31, 64, 72, -57, 40, 72, 3, 20, 50, 38, -68, -51, -16, 51, 36, -68, -43, -36, 55, 72, -54, -64, 34, 71, -48, 3, 52, 52, -51, 14, 26, -46, -19, 17, 35, -73, -47, 14, -39, -61, -37, -47, 10, -10, -7, 36, 25, 48, 15, -6, -19, -70, 31, 8, 38, -34, -46, -18, 62, 24, 72, 67, -71, 14, 72, 64, -31, 16, 15, -7, 68, 9, -39, 18, -41, 45, -39, 13, 37, -36, -45, 46, -66, -11, 21, -67, -33, 3, -66, -68, -29, 20, 63, 31, 57, -59, 4, -16, -48, -69, -15, 9, -72, -29, 34, 44, -3, 1, 47, -22, 49, 63, 65, 4, 50, 50, -45, 29, 55, 53, -53, -57, 33, 29, 26, 49, 25, -69, 45, -54, -69, 24, 22, -4, -47, 22, -19, 55, 7, 45, -58, -6, 56, -46, 33, -39, 13, 61, -65, 0, 10, -45, 33, 23, -60, -53, 70, -61, 24, 33, -31, -47, -29, -11, -58, 72, -11, 2, -32, 15, -43, -48, 37, 15, 73, 49, 9, 19, 35, 11, 15, -9, -42, 43, 44, -18, -28, -13, -70, -2, 68, 42, 3, 37, -17, 42, -36, -27, 42, -46, 50, -59, 0, 2, -38, 45, -26, -56, 1, -6, -57, 55, -69, 13, 58, 30, -3, -48, -41, -17, 2, -46, 60, 42, 55, 52, 6, 54, -13, -35, 55, -31, 23, 55, -10, 4, 69, 37, 11, -29, -65, 65, 1, 48, -46, 3, -29, 40, -61, 60, -38, -3, -46, 68, 11, 66, -60, -6, 12, -4, -11, -63, 27, 28, 50, -51, 47, -47, 31, 39, 10, -13, -5, -20, 42, -48, -22, 65, -11, 57, 61, 70, -34, -23, 56, -1, -6, 61, 70, 29, 11, -18, 56, -20, 34, 64, -6, -70, 11, -31, 26, 58, -38, 16, 62, 73, 8, -43, 64, 49, -46, -64, 3, 37, 4, -20, 73, 20, -30, -59, 55, 65, 0, 33, 23, -22, -11, -27, -27, -34, 24, 39, 23, 13, -70, 14, -49, -25, 21, -31, -2, 32, -44, 48, 70, 21, 30, -30, 40, 40, 36, -27, 52, -22, 24, 24, 20, 37, 2, 23, -39, -49, 55, -66, -44, -32, -26, 23, 11, 29, 62, -5, -34, 17, -10, -24, -73, -10, -28, 63, -54, -38, 68, -22, -33, -40, 17, -56, 23, 21, -51, -2, -41, -63, 68, -2, 33, -22, -45, 50, -10, -42, -54, -47, 70, 58, 17, 65, -47, 49, -21, -51, 7, 34, -14, 29, -40, -55, -10, 58, 61, 16, -7, -49, -11, -71, 53, 19, 38, 6, -24, -12, -9, -17, 15, -3, -58, 12, -6, 72, -62, -60, -10, -40, -53, 72, -26, 49, 33, 34, 29, -65, -52, -58, -64, -16, -70, 45, -60, 42, 73, -56, 57, 56, -52, -54, -11, -48, 39, -25, -3, -6, -49, 55, -21, -71, 67, -51, 45, -5, -70, -23, -14, -71, -46, -68, -65, -13, -8, 0, 45, -65, -60, 45, 6, -63, -25, -31, 21, -58, -69, 60, 64, -41, -68, 49, 72, -36, 55, 19, -44, -30, 7, -19, -13, -28, -54, -43, -63, -38, 50, 36, -48, -8, -5, 65, -22, 17, -32, 66, -73, -43, 15, -6, 12, 26, -24, 18, 31, -18, -38, -56, 11, 50, 60, 33, -65, 14, 0, -18, -40, -51, 44, 51, 68, 46, 36, -9, 35, -64, 46, 60, -12, -50, 4, 12, 21, 0, 23, 60, 67, 63, 5, 47, 29, -28, -55, 40, -13, -15, 69, -41, 14, -50, -66, 12, 63, -73, -32, -10, -43, -17, 35, 68, 63, 14, -21, -61, 45, -6, -19, 36, -11, 31, 67, 37, -34, -62, 34, -49, -62, 5, 32, 35, -10, -21, -20, -47, -24, -38, -35, 18, 53, -45, 55, -64, -40, -64, 42, -17, -34, -68, -60, 24, -20, 68, -26, 32, 10, 45, 61, 46, -56, 35, 70, -4, -62, 38, 59, -43, -71, 46, 1, 54, -39, -67, -66, -13, -72, 26, -45, -14, 72, -16, 14, -45, -52, 22, 22, 2, 12, 11, 27, 9, -22, 56, -45, 51, -72, -61, 5, -14, 15, 53, 59, 0, -17, -47, 4, -31, 53, -68, 33, -42, 26, 38, -33, -51, -68, -30, 22, 18, -2, 46, -64, -19, 70, -23, 63, 39, 63, -30, 62, 19, 47, -18, -35, 35, 31, 33, 25, -33, 5, 21, 12, -9, -2, -62, 50, -12, 42, 12, -59, -61, 8, -3, -21, -49, -12, -9, -59, -10, -57, 7, -68, 71, -19, 15, -26, -36, -37, -25, -58, -50, 28, 8, 3, -25, -51, 51, 10, -16, 41, 47, -6, -70, 33, 20, 20, 54, 66, 72, 3, 1, -59, -57, -23, 53, -4, 69, -40, -33, 30, 30, 24, 39, 31, -16, 16, -10, 14, 10, -68, 61, 26, 52, -65, -58, -70, -50, 33, 15, 67, -11, -45, 48, -49, -8, -12, 9, 5, 52, 38, 53, -68, 0, 30, 46, -66, -21, 13, 51, 43, 2, -65, 23, -51, -22, -61, 60, -68, 38, -27, 59, -34, 54, 53, 23, -7, -5, -35, 40, 62, 6, -69, -50, -38, -56, 11, 72, -65, 71, 26, -31, 37, 37, -55, 29, 47, 10, 32, 4, -15, -4, -47, 61, -13, -37, -54, 56, -5, 12, -12, 63, -49, -17, -10, 23, 68, 15, -58, -56, -71, -29, 49, 46, -21, -34, 13, 52, 65, 8, -52, 33, -9, 57, -38, 71, -27, 69, 65, -63, 73, -21, -5, 51, -34, -40, 12, 36, 48, 28, -12, 45, 31, -72, -10, 10, 61, -15, -3, 56, 70, -53, 29, 15, 4, 33, 51, -20, -30, 25, -7, -25, 44, -69, 10, -25, 8, -72, 32, -20, 53, 66, -62, 8, -62, -71, -8, -7, -5, -16, -17, -52, -61, 47, -48, 39, 42, -29, 14, -53, -51, -46, -57, 16, 39, 31, 62, 63, 63, 44, -15, 40, -33, 29, -64, 1, 28, -62, 54, 24, 31, -48, -36, -23, 66, -32, 62, -21, 29, 38, -59, -43, -69, 36, 65, -35, 38, -62, 70, 29, -56, -61, 20, 37, -13, 59, -9, -55, -54, -38, -4, -57, -23, -40, -36, 71, 41, -35, -31, -52, 20, -15, -24, -47, 36, 63, -56, -7, -11, -53, -70, -19, -23, 61, 55, -22, -3, -24, 52, -2, -17, -70, 41, -62, -62, -1, 6, -56, 15, 63, -29, 14, -4, 18, 29, 49, -13, 57, 71, 14, -54, -35, -55, -36, -23, 44, 23, -39, 19, -26, -69, 19, 69, -29, 25, 64, 5, 27, 57, 32, 34, -21, 35, -56, 36, -32, -58, -11, -18, -69, 11, 44, -57, -11, -54, 19, -24, -33, -50, 10, -32, -55, -18, -52, -55, -58, 41, 5, 54, 34, -59, 1, 30, 18, 38, -47, -57, 36, -8, 20, 20, 21, -54, 29, 19, 49, 5, -34, 44, 17, -66, -2, 24, -11, 61, -27, -64, -11, -30, -41, 16, -44, 48, -68, 37, 22, 3, 42, 70, 0, -24, 58, 66, 7, -67, 5, 54, 1, -56, 33, 28, 62, -65, -30, -13, -69, 34, 7, 48, -6, -34, 25, 15, -64, -3, -36, 1, -44, -4, 57, 29, -31, -42, 36, 38, 42, -18, 67, 11, -16, -43, 59, 69, 71, 49, -68, -16, -18, 27, 50, -21, -10, 26, 11, -18, 62, 45, 48, 0, -14, 10, 4, -38, 21, 40, -73, 7, 6, -34, -54, -11, 50, 7, 57, -34, -62, -20, -47, 59, -10, -9, 69, -57, -13, -17, 66, -54, -13, 51, 29, 16, -38, 29, -31, 1, 62, -68, 56, 17, 62, -54, 41, 1, 39, -45, 67, 31, 11, -32, -56, -50, 10, 64, -10, 19, 67, -2, -7, -40, -16, 63, -17, 0, -55, 8, 0, 6, 65, -72, -56, 34, -19, 22, -65, -53, 25, 44, -72, 55, 11, 32, 40, 22, -25, -59, 17, -5, -71, -60, 70, -23, 13, -40, -61, 16, 5, -22, -63, 27, 63, 54, -33, -36, -58, -19, -35, 63, -39, 3, -44, -28, 63, 4, -16, 15, 68, 12, 23, 43, -12, -29, 69, -8, 62, 61, 71, 26, -72, -17, -21, -36, 55, 55, 8, 21, -9, 28, -20, 40, 34, -46, -23, -41, 5, -44, -56, 4, 64, 47, -68, -30, -1, -6, -31, 13, -58, 68, 2, -27, -43, -67, 58, 60, 37, 71, 23, 5, 23, 55, -30, 10, -35, 44, -32, -69, 8, 45, 24, -73, -5, -60, 0, -14, 63, 44, 28, 58, 5, -5, 54, -43, -73, -43, 26, -2, -10, 38, 8, -14, -66, 9, -56, -57, -55, -25, -30, 14, 20, -15, 28, -6, -7, 19, 16, 36, 72, -58, -24, -52, 18, -49, 24, 56, -73, 45, -36, 2, 55, -31, 21, -29, -41, 38, 46, 69, 11, 40, -66, -17, 64, 37, 28, -3, 28, 16, 6, -68, -28, -20, 37, -1, -15, 21, 14, -61, 11, -17, -21, -18, 12, 38, -67, 0, -68, -66, 34, -9, -6, -42, 4, -10, -8, 38, 65, 62, 21, -50, 20, 42, 21, 63, 71, -31, -27, -37, -70, 6, 4, -29, -34, -36, -10, 72, 20, -28, 16, 57, -24, 18, 67, -14, 69, 67, 46, -42, 24, -70, -18, 35, 59, 2, -66, 48, 39, 59, -16, -34, 20, -61, -10, 73, -15, -1, 26, 11, 8, 0, 53, -69, 44, 47, -41, -5, 43, -29, 34, 26, 9, -14, -31, -69, 71, 62, 34, 6, 13, -63, 63, 12, 53, 13, -29, 43, 41, 32, -3, 50, -43, 30, 33, 64, -66, -32, -17, -44, 44, -60, 34, 68, -15, -11, -71, -68, -4, -42, -59, -12, 67, -40, -38, -30, -52, 59, -51, 39, 27, -30, 11, -51, 70, 53, -63, 64, -7, -58, -34, 35, 67, 36, 25, 44, 15, -4, -19, 4, 26, 16, 2, 65, -52, -44, 29, -2, -50, -69, 23, 25}

#define GROUP_CONV2D_KERNEL_0_SHIFT (10)

#define GROUP_CONV2D_BIAS_0 {91, -55, -64, -95, 65, -93, 0, 18, -66, -66, 12, -21, 100, 56, -93, 60}

#define GROUP_CONV2D_BIAS_0_SHIFT (8)


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define GROUP_CONV2D_OUTPUT_SHIFT 5

/* bias shift and output shift for each layer */
#define GROUP_CONV2D_OUTPUT_RSHIFT (INPUT_1_OUTPUT_SHIFT+GROUP_CONV2D_KERNEL_0_SHIFT-GROUP_CONV2D_OUTPUT_SHIFT)
#define GROUP_CONV2D_BIAS_LSHIFT   (INPUT_1_OUTPUT_SHIFT+GROUP_CONV2D_KERNEL_0_SHIFT-GROUP_CONV2D_BIAS_0_SHIFT)
#if GROUP_CONV2D_OUTPUT_RSHIFT < 0
#error GROUP_CONV2D_OUTPUT_RSHIFT must be bigger than 0
#endif
#if GROUP_CONV2D_BIAS_LSHIFT < 0
#error GROUP_CONV2D_BIAS_LSHIFT must be bigger than 0
#endif

/* weights for each layer */
static const int8_t group_conv2d_weights[] = GROUP_CONV2D_KERNEL_0;
static const nnom_weight_t group_conv2d_w = { (const void*)group_conv2d_weights, GROUP_CONV2D_OUTPUT_RSHIFT};
static const int8_t group_conv2d_bias[] = GROUP_CONV2D_BIAS_0;
static const nnom_bias_t group_conv2d_b = { (const void*)group_conv2d_bias, GROUP_CONV2D_BIAS_LSHIFT};

/* nnom model */
static int8_t nnom_input_data[16384];
static int8_t nnom_output_data[16384];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 16), nnom_input_data);
	layer[1] = model.hook(GroupConv2D(16, 2, kernel(7, 7), stride(1, 1), dilation(1, 1), PADDING_SAME, &group_conv2d_w, &group_conv2d_b), layer[0]);
	layer[2] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}