# GARCH-variants
Compare the fit of many GARCH models to return data. Most are from the [arch](https://pypi.org/project/arch/) package, but a few are coded by hand.

`python xgarch_skew_data.py --optimizer bfgs` gives

```
optimizer: bfgs
loaded price data from prices.csv with 5588 rows and 4 columns
date range: 2003-04-14 to 2025-06-27
return scaling factor: 100.0
demean returns: yes
return type: log

==== return summary (annualized, obs_year = 252) ====
symbol           ann_return        ann_vol        skew    kurtosis         min         max
SPY                 10.5856        18.7741     -0.3046     17.8257    -11.5887     13.5577
EFA                  7.4403        20.8488     -0.3659     16.6890    -11.8369     14.7451
EEM                  8.4783        27.2377      0.0245     18.9021    -17.6334     20.5141
TLT                  3.3566        14.5340     -0.0062      6.3784     -6.9010      7.2502

correlations
                     SPY         EFA         EEM         TLT
SPY                1.000       0.880       0.815      -0.301
EFA                0.880       1.000       0.868      -0.277
EEM                0.815       0.868       1.000      -0.262
TLT               -0.301      -0.277      -0.262       1.000

==== symbol: SPY ====
price observations: 5588, log returns used: 5587
         n           mean             sd           skew    ex_kurtosis            min            max
      5587         0.0420         1.1827        -0.3046        14.8257       -11.5887        13.5577

autocorrelations (lag 1-5)
   lag      returns    |returns|    returns^2
     1       -0.101        0.315        0.274
     2       -0.009        0.402        0.452
     3        0.007        0.339        0.250
     4       -0.031        0.357        0.293
     5       -0.014        0.358        0.303

model                               uncond_sd                  mu               omega               alpha                beta               gamma               shift                skew                 dof              loglik            n_params                AICC                 BIC         loglik_rank           AICC_rank            BIC_rank
nagarch_normal                       1.582801           -0.017132            0.029359            0.103902            0.751761            1.129764                  NA            0.000000                  NA        -7197.478490                   5        14404.967731        14438.097969                   9                   9                   9
nagarch_student_t                    1.855043            0.012896            0.022875            0.100350            0.730317            1.273252                  NA            0.000000            6.529618        -7065.558322                   6        14143.131697        14182.885830                   3                   3                   3
nagarch_skew_student_t               2.166183           -0.014865            0.025799            0.094620            0.723177            1.366572                  NA           -0.186629            7.379929        -7014.568151                   7        14043.156377        14089.533686                   1                   1                   1
garch_student_t                      1.527093            0.053405            0.019024            0.136546            0.855296                  NA                  NA                  NA            5.737347        -7209.033663                   5        14428.078076        14461.208314                  10                  10                  10
garch_skew_student_t                 1.289603            0.031288            0.017830            0.131060            0.858218                  NA                  NA           -0.125510            6.437155        -7184.708783                   6        14381.432620        14421.186753                   8                   8                   8
garch_normal                         1.078115            0.034971            0.028165            0.131385            0.844384                  NA                  NA                  NA                  NA        -7363.275038                   4        14734.557242        14761.062867                  15                  15                  15
gjr_student_t                        1.023070            0.026038            0.022660            0.000000            0.861182            0.234337                  NA                  NA            6.192412        -7114.279063                   6        14240.573179        14280.327312                   7                   7                   7
gjr_skew_student_t                   1.123414           -0.002190            0.023696            0.000000            0.860988            0.240472                  NA           -0.164444            6.880928        -7074.136283                   7        14162.292641        14208.669950                   4                   4                   4
gjr_normal                           0.971750           -0.001575            0.027587            0.006034            0.866809            0.195885                  NA                  NA                  NA        -7263.458073                   5        14536.926896        14570.057134                  14                  14                  14
egarch_student_t                           NA            0.023023           -0.005337            0.159210            0.974890                  NA                  NA                  NA            6.198433        -7100.512046                   6        14213.039146        14252.793279                   6                   6                   6
egarch_skew_student_t                      NA           -0.006113           -0.000615            0.157101            0.971862                  NA                  NA           -0.173259            6.892514        -7056.540101                   7        14127.100278        14173.477587                   2                   2                   2
egarch_normal                              NA           -0.001658           -0.001067            0.172719            0.966835                  NA                  NA                  NA                  NA        -7251.787460                   5        14513.585671        14546.715909                  12                  12                  13
igarch_student_t                           NA            0.049843                  NA            0.092163            0.907837                  NA                  NA            0.000000            6.435398        -7258.635266                   3        14523.274832        14543.155126                  13                  13                  12
igarch_normal                              NA            0.021478                  NA            0.073242            0.926758                  NA                  NA            0.000000                  NA        -7495.266378                   2        14994.534905        15007.789152                  16                  16                  16
st_student_t                         0.389585            0.013931            0.025111            0.038500            0.764564            0.062982            1.181592            0.000000            6.392373        -7076.697536                   7        14167.415147        14213.792456                   5                   5                   5
st_normal                            0.468420           -0.017178            0.030919            0.038262            0.786914            0.067821            0.999189            0.000000                  NA        -7211.684637                   6        14435.384327        14475.138460                  11                  11                  11
constant_vol                         1.182654            0.000000            1.398672                  NA                  NA                  NA                  NA                  NA                  NA        -8864.893010                   1        17731.786737        17738.414218                  17                  17                  17

loglik selects
model                                 loglik               diff
nagarch_skew_student_t           7014.568151           0.000000  best
egarch_skew_student_t            7056.540101          41.971950
nagarch_student_t                7065.558322          50.990171
gjr_skew_student_t               7074.136283          59.568132
st_student_t                     7076.697536          62.129385
egarch_student_t                 7100.512046          85.943895
gjr_student_t                    7114.279063          99.710912
garch_skew_student_t             7184.708783         170.140632
nagarch_normal                   7197.478490         182.910339
garch_student_t                  7209.033663         194.465512
st_normal                        7211.684637         197.116486
egarch_normal                    7251.787460         237.219309
igarch_student_t                 7258.635266         244.067116
gjr_normal                       7263.458073         248.889922
garch_normal                     7363.275038         348.706887
igarch_normal                    7495.266378         480.698227
constant_vol                     8864.893010        1850.324859

AICC selects
model                                   AICC               diff
nagarch_skew_student_t          14043.156377           0.000000  best
egarch_skew_student_t           14127.100278          83.943901
nagarch_student_t               14143.131697          99.975320
gjr_skew_student_t              14162.292641         119.136264
st_student_t                    14167.415147         124.258770
egarch_student_t                14213.039146         169.882769
gjr_student_t                   14240.573179         197.416802
garch_skew_student_t            14381.432620         338.276243
nagarch_normal                  14404.967731         361.811354
garch_student_t                 14428.078076         384.921699
st_normal                       14435.384327         392.227950
egarch_normal                   14513.585671         470.429294
igarch_student_t                14523.274832         480.118455
gjr_normal                      14536.926896         493.770519
garch_normal                    14734.557242         691.400865
igarch_normal                   14994.534905         951.378528
constant_vol                    17731.786737        3688.630360

BIC selects
model                                    BIC               diff
nagarch_skew_student_t          14089.533686           0.000000  best
egarch_skew_student_t           14173.477587          83.943901
nagarch_student_t               14182.885830          93.352144
gjr_skew_student_t              14208.669950         119.136264
st_student_t                    14213.792456         124.258770
egarch_student_t                14252.793279         163.259593
gjr_student_t                   14280.327312         190.793625
garch_skew_student_t            14421.186753         331.653067
nagarch_normal                  14438.097969         348.564283
garch_student_t                 14461.208314         371.674628
st_normal                       14475.138460         385.604774
igarch_student_t                14543.155126         453.621440
egarch_normal                   14546.715909         457.182223
gjr_normal                      14570.057134         480.523448
garch_normal                    14761.062867         671.529181
igarch_normal                   15007.789152         918.255466
constant_vol                    17738.414218        3648.880532

standardized residual stats
model                              n           mean             sd           skew    ex_kurtosis            min            max
nagarch_normal                  5587         0.0111         1.0003        -0.6542         2.1424        -6.6574         3.7428
nagarch_student_t               5587        -0.0280         1.0018        -0.6836         2.4444        -7.4416         3.8546
nagarch_skew_student_t          5587         0.0070         1.0064        -0.6824         2.4401        -7.4436         3.9367
garch_student_t                 5587        -0.0681         0.9938        -0.6541         2.3752        -8.0479         3.7110
garch_skew_student_t            5587        -0.0420         1.0069        -0.6579         2.4175        -8.1738         3.7933
garch_normal                    5587        -0.0448         0.9991        -0.6093         2.0928        -7.5834         3.8588
gjr_student_t                   5587        -0.0408         1.0009        -0.7201         2.8391        -7.5785         3.4469
gjr_skew_student_t              5587        -0.0058         1.0050        -0.7213         2.8490        -7.5995         3.4831
gjr_normal                      5587        -0.0051         1.0002        -0.6743         2.4383        -7.2765         3.5417
egarch_student_t                5587        -0.0396         1.0023        -0.7429         3.0016        -8.3881         3.7917
egarch_skew_student_t           5587        -0.0035         1.0057        -0.7435         3.0228        -8.3229         3.8507
egarch_normal                   5587        -0.0073         1.0002        -0.6824         2.4334        -7.3003         3.7127
igarch_student_t                5587        -0.0724         1.0819        -0.8351         3.9281       -10.2724         4.1533
igarch_normal                   5587        -0.0336         1.0678        -0.8042         3.7342        -9.9272         4.4456
st_student_t                    5587        -0.0285         0.9987        -0.6988         2.5777        -7.8037         3.8008
st_normal                       5587         0.0118         0.9993        -0.6647         2.2459        -6.9998         3.7370
constant_vol                    5587         0.0000         1.0000        -0.3046        14.8257        -9.8344        11.4283

conditional sd stats
model                              n           mean             sd           skew    ex_kurtosis            min            max
nagarch_normal                  5587         0.9975         0.6673         3.8961        22.3285         0.4062         7.4640
nagarch_student_t               5587         1.0100         0.6982         3.7618        21.0228         0.3622         7.5716
nagarch_skew_student_t          5587         1.0035         0.6938         3.7882        21.1751         0.3705         7.4575
garch_student_t                 5587         1.0203         0.6432         3.8247        20.9411         0.3980         6.9241
garch_skew_student_t            5587         1.0083         0.6353         3.8198        20.8386         0.3914         6.7979
garch_normal                    5587         1.0017         0.6001         3.9591        22.3522         0.4508         6.7084
gjr_student_t                   5587         1.0121         0.6723         3.7250        20.7795         0.4217         8.0266
gjr_skew_student_t              5587         1.0097         0.6774         3.7545        21.0403         0.4266         8.1037
gjr_normal                      5587         0.9956         0.6263         3.8339        21.5870         0.4695         7.5096
egarch_student_t                5587         0.9973         0.5861         3.0771        15.4875         0.2881         6.6603
egarch_skew_student_t           5587         0.9932         0.5798         3.0495        15.3005         0.2907         6.6214
egarch_normal                   5587         0.9819         0.5375         3.0744        15.5880         0.3213         6.3117
igarch_student_t                5587         0.9795         0.6657         3.4298        16.5721         0.2217         6.0568
igarch_normal                   5587         0.9873         0.6525         3.3356        15.3300         0.2564         5.5913
st_student_t                    5587         1.0112         0.6898         3.8349        21.9118         0.3814         7.7000
st_normal                       5587         0.9962         0.6543         3.9352        22.7898         0.4228         7.4947
constant_vol                    5587         1.1827         0.0000         1.0000        -2.0000         1.1827         1.1827

==== symbol: EFA ====
price observations: 5588, log returns used: 5587
         n           mean             sd           skew    ex_kurtosis            min            max
      5587         0.0295         1.3134        -0.3659        13.6890       -11.8369        14.7451

autocorrelations (lag 1-5)
   lag      returns    |returns|    returns^2
     1       -0.107        0.314        0.257
     2        0.009        0.370        0.398
     3        0.010        0.315        0.239
     4       -0.015        0.308        0.248
     5       -0.008        0.324        0.312

model                               uncond_sd                  mu               omega               alpha                beta               gamma               shift                skew                 dof              loglik            n_params                AICC                 BIC         loglik_rank           AICC_rank            BIC_rank
nagarch_normal                       1.512390           -0.012825            0.024036            0.088710            0.851271            0.747073                  NA            0.000000                  NA        -8043.002560                   5        16096.015871        16129.146109                  11                  11                  11
nagarch_student_t                    1.448514            0.010851            0.020263            0.080462            0.850596            0.858364                  NA            0.000000            7.439694        -7935.807926                   6        15883.630906        15923.385039                   4                   4                   4
nagarch_skew_student_t               1.699032           -0.016379            0.022203            0.079816            0.846358            0.910273                  NA           -0.160299            7.797749        -7899.558577                   7        15813.137229        15859.514538                   1                   1                   1
garch_student_t                      1.352863            0.042568            0.020644            0.104956            0.883765                  NA                  NA                  NA            6.820958        -7998.978839                   5        16007.968429        16041.098667                   9                   9                   9
garch_skew_student_t                 1.293155            0.020025            0.019533            0.102315            0.886004                  NA                  NA           -0.131552            7.336419        -7973.880907                   6        15959.776867        15999.531000                   8                   8                   8
garch_normal                         1.332880            0.026220            0.023847            0.112478            0.874099                  NA                  NA                  NA                  NA        -8119.074598                   4        16246.156363        16272.661988                  15                  15                  15
gjr_student_t                        1.080969            0.022270            0.022884            0.020555            0.893044            0.133633                  NA                  NA            7.326469        -7955.969017                   6        15923.953088        15963.707221                   7                   7                   7
gjr_skew_student_t                   1.137502           -0.002367            0.023748            0.019767            0.893040            0.137679                  NA           -0.146243            7.755245        -7925.322329                   7        15864.664733        15911.042042                   3                   3                   3
gjr_normal                           1.164891           -0.000701            0.023939            0.030447            0.888756            0.126310                  NA                  NA                  NA        -8068.715424                   5        16147.441599        16180.571837                  14                  14                  14
egarch_student_t                           NA            0.014561            0.003446            0.150334            0.982237                  NA                  NA                  NA            7.570381        -7937.535004                   6        15887.085061        15926.839194                   5                   5                   5
egarch_skew_student_t                      NA           -0.009100            0.006051            0.149633            0.980906                  NA                  NA           -0.151883            7.957244        -7904.581478                   7        15823.183031        15869.560340                   2                   2                   2
egarch_normal                              NA           -0.008165            0.007083            0.167912            0.979303                  NA                  NA                  NA                  NA        -8050.493153                   5        16110.997056        16144.127294                  13                  13                  12
igarch_student_t                           NA            0.040018                  NA            0.072538            0.927462                  NA                  NA            0.000000            7.553158        -8034.401225                   3        16074.806749        16094.687044                  10                  10                  10
igarch_normal                              NA            0.019665                  NA            0.075793            0.924207                  NA                  NA            0.000000                  NA        -8183.029156                   2        16370.060462        16383.314708                  16                  16                  16
st_student_t                         0.597152            0.012462            0.023155            0.078989            0.851536            0.009082            0.750558            0.000000            7.485696        -7940.921640                   7        15895.863354        15942.240663                   6                   6                   6
st_normal                            0.674271           -0.012124            0.025634            0.093931            0.849682            0.000011            0.692969            0.000000                  NA        -8046.365123                   6        16104.745300        16144.499432                  12                  12                  13
constant_vol                         1.313351            0.000000            1.724890                  NA                  NA                  NA                  NA                  NA                  NA        -9450.522597                   1        18903.045910        18909.673391                  17                  17                  17

loglik selects
model                                 loglik               diff
nagarch_skew_student_t           7899.558577           0.000000  best
egarch_skew_student_t            7904.581478           5.022901
gjr_skew_student_t               7925.322329          25.763752
nagarch_student_t                7935.807926          36.249349
egarch_student_t                 7937.535004          37.976427
st_student_t                     7940.921640          41.363062
gjr_student_t                    7955.969017          56.410440
garch_skew_student_t             7973.880907          74.322330
garch_student_t                  7998.978839          99.420262
igarch_student_t                 8034.401225         134.842648
nagarch_normal                   8043.002560         143.443983
st_normal                        8046.365123         146.806546
egarch_normal                    8050.493153         150.934575
gjr_normal                       8068.715424         169.156847
garch_normal                     8119.074598         219.516021
igarch_normal                    8183.029156         283.470579
constant_vol                     9450.522597        1550.964020

AICC selects
model                                   AICC               diff
nagarch_skew_student_t          15813.137229           0.000000  best
egarch_skew_student_t           15823.183031          10.045802
gjr_skew_student_t              15864.664733          51.527504
nagarch_student_t               15883.630906          70.493676
egarch_student_t                15887.085061          73.947832
st_student_t                    15895.863354          82.726125
gjr_student_t                   15923.953088         110.815859
garch_skew_student_t            15959.776867         146.639638
garch_student_t                 16007.968429         194.831200
igarch_student_t                16074.806749         261.669520
nagarch_normal                  16096.015871         282.878642
st_normal                       16104.745300         291.608070
egarch_normal                   16110.997056         297.859826
gjr_normal                      16147.441599         334.304370
garch_normal                    16246.156363         433.019133
igarch_normal                   16370.060462         556.923232
constant_vol                    18903.045910        3089.908680

BIC selects
model                                    BIC               diff
nagarch_skew_student_t          15859.514538           0.000000  best
egarch_skew_student_t           15869.560340          10.045802
gjr_skew_student_t              15911.042042          51.527504
nagarch_student_t               15923.385039          63.870500
egarch_student_t                15926.839194          67.324656
st_student_t                    15942.240663          82.726125
gjr_student_t                   15963.707221         104.192683
garch_skew_student_t            15999.531000         140.016461
garch_student_t                 16041.098667         181.584129
igarch_student_t                16094.687044         235.172505
nagarch_normal                  16129.146109         269.631571
egarch_normal                   16144.127294         284.612755
st_normal                       16144.499432         284.984894
gjr_normal                      16180.571837         321.057299
garch_normal                    16272.661988         413.147449
igarch_normal                   16383.314708         523.800170
constant_vol                    18909.673391        3050.158853

standardized residual stats
model                              n           mean             sd           skew    ex_kurtosis            min            max
nagarch_normal                  5587         0.0065         0.9996        -0.5092         1.9886        -7.5506         4.6535
nagarch_student_t               5587        -0.0189         0.9988        -0.5195         2.0997        -7.8317         4.6578
nagarch_skew_student_t          5587         0.0096         0.9977        -0.5218         2.1284        -7.8694         4.6911
garch_student_t                 5587        -0.0477         0.9988        -0.5017         1.8467        -6.2510         4.5205
garch_skew_student_t            5587        -0.0245         1.0048        -0.5052         1.8662        -6.3175         4.5781
garch_normal                    5587        -0.0306         0.9996        -0.4973         1.8293        -6.2225         4.5395
gjr_student_t                   5587        -0.0289         1.0021        -0.5072         2.0364        -7.4713         4.6028
gjr_skew_student_t              5587        -0.0031         1.0032        -0.5075         2.0394        -7.4539         4.6467
gjr_normal                      5587        -0.0046         1.0002        -0.5031         1.9740        -7.2435         4.6326
egarch_student_t                5587        -0.0214         1.0038        -0.5343         2.2807        -8.0143         4.6721
egarch_skew_student_t           5587         0.0036         1.0031        -0.5335         2.2870        -8.0079         4.7234
egarch_normal                   5587         0.0029         1.0002        -0.5242         2.1890        -7.7638         4.6830
igarch_student_t                5587        -0.0495         1.0571        -0.5382         2.1880        -7.0729         5.4180
igarch_normal                   5587        -0.0269         1.0597        -0.5344         2.1716        -7.0261         5.4843
st_student_t                    5587        -0.0197         1.0003        -0.5076         2.0261        -7.7003         4.5846
st_normal                       5587         0.0064         0.9989        -0.5001         1.9325        -7.4125         4.6212
constant_vol                    5587        -0.0000         1.0000        -0.3659        13.6890        -9.0352        11.2046

conditional sd stats
model                              n           mean             sd           skew    ex_kurtosis            min            max
nagarch_normal                  5587         1.1335         0.6870         3.9586        21.9066         0.4800         7.2316
nagarch_student_t               5587         1.1372         0.6917         3.8887        21.1896         0.4528         7.1410
nagarch_skew_student_t          5587         1.1390         0.6969         3.9087        21.3699         0.4590         7.2391
garch_student_t                 5587         1.1407         0.6605         3.8493        20.4941         0.4826         6.9332
garch_skew_student_t            5587         1.1349         0.6576         3.8407        20.3869         0.4776         6.8876
garch_normal                    5587         1.1393         0.6584         3.8921        20.9947         0.4925         6.9875
gjr_student_t                   5587         1.1335         0.6722         3.7527        19.2861         0.5075         6.8571
gjr_skew_student_t              5587         1.1330         0.6755         3.7700        19.4229         0.5112         6.9075
gjr_normal                      5587         1.1357         0.6761         3.7924        19.6769         0.5066         6.8936
egarch_student_t                5587         1.1183         0.5946         3.2752        16.1212         0.3847         5.9395
egarch_skew_student_t           5587         1.1185         0.5926         3.2612        16.0275         0.3861         5.9225
egarch_normal                   5587         1.1211         0.5940         3.2924        16.3516         0.3882         5.9262
igarch_student_t                5587         1.1134         0.6995         3.4471        16.4960         0.2849         6.6514
igarch_normal                   5587         1.1117         0.7012         3.4686        16.7320         0.2830         6.7129
st_student_t                    5587         1.1335         0.6835         3.9287        21.6044         0.4732         7.1574
st_normal                       5587         1.1342         0.6864         3.9834        22.1792         0.4901         7.3017
constant_vol                    5587         1.3134         0.0000         1.0000        -2.0000         1.3134         1.3134

==== symbol: EEM ====
price observations: 5588, log returns used: 5587
         n           mean             sd           skew    ex_kurtosis            min            max
      5587         0.0336         1.7158         0.0245        15.9021       -17.6334        20.5141

autocorrelations (lag 1-5)
   lag      returns    |returns|    returns^2
     1       -0.108        0.302        0.222
     2       -0.032        0.392        0.486
     3        0.031        0.306        0.184
     4       -0.036        0.338        0.321
     5       -0.006        0.358        0.337

model                               uncond_sd                  mu               omega               alpha                beta               gamma               shift                skew                 dof              loglik            n_params                AICC                 BIC         loglik_rank           AICC_rank            BIC_rank
nagarch_normal                       1.570881           -0.012690            0.048484            0.082245            0.865689            0.627834                  NA            0.000000                  NA        -9553.505456                   5        19117.021663        19150.151900                  10                  10                  10
nagarch_student_t                    1.534173            0.010198            0.049197            0.085188            0.856238            0.665000                  NA            0.000000            9.880137        -9491.081995                   6        18994.179044        19033.933177                   4                   4                   4
nagarch_skew_student_t               1.608426           -0.013004            0.048995            0.083430            0.858199            0.687482                  NA           -0.130536           10.151114        -9467.949008                   7        18949.918092        18996.295401                   1                   1                   1
garch_student_t                      1.525795            0.043945            0.046396            0.100932            0.879139                  NA                  NA                  NA            9.208456        -9534.244640                   5        19078.500031        19111.630269                   9                   9                   9
garch_skew_student_t                 1.508724            0.022710            0.043240            0.098438            0.882566                  NA                  NA           -0.117636            9.729874        -9515.142722                   6        19042.300497        19082.054630                   8                   8                   8
garch_normal                         1.531037            0.026350            0.044855            0.098122            0.882742                  NA                  NA                  NA                  NA        -9603.501097                   4        19215.009360        19241.514985                  15                  15                  15
gjr_student_t                        1.401683            0.014386            0.049067            0.028561            0.887566            0.117797                  NA                  NA            9.859902        -9499.237140                   6        19010.489334        19050.243466                   6                   6                   6
gjr_skew_student_t                   1.444635           -0.009187            0.048186            0.026062            0.890859            0.119982                  NA           -0.129574           10.178242        -9476.328192                   7        18966.676459        19013.053768                   2                   2                   2
gjr_normal                           1.427812           -0.009477            0.046567            0.028122            0.895542            0.106986                  NA                  NA                  NA        -9561.629080                   5        19133.268911        19166.399149                  12                  12                  12
egarch_student_t                           NA            0.015230            0.013774            0.166134            0.981173                  NA                  NA                  NA            9.473700        -9509.352788                   6        19030.720630        19070.474763                   7                   7                   7
egarch_skew_student_t                      NA           -0.008504            0.015194            0.163888            0.981228                  NA                  NA           -0.129528            9.766038        -9486.414747                   7        18986.849569        19033.226878                   3                   3                   3
egarch_normal                              NA           -0.009396            0.015224            0.158492            0.981326                  NA                  NA                  NA                  NA        -9576.848928                   5        19163.708607        19196.838845                  14                  14                  14
igarch_student_t                           NA            0.041122                  NA            0.075280            0.924720                  NA                  NA            0.000000            9.846267        -9576.571828                   3        19159.147955        19179.028250                  13                  13                  13
igarch_normal                              NA            0.017057                  NA            0.072321            0.927679                  NA                  NA            0.000000                  NA        -9658.535868                   2        19321.073884        19334.328131                  16                  16                  16
st_student_t                         0.937624            0.010245            0.050467            0.072191            0.860530            0.019750            0.572469            0.000000            9.860965        -9491.623434                   7        18997.266944        19043.644253                   5                   5                   5
st_normal                            0.982984           -0.013061            0.049363            0.074476            0.868344            0.012188            0.564233            0.000000                  NA        -9553.979401                   6        19119.973856        19159.727989                  11                  11                  11
constant_vol                         1.715815            0.000000            2.944023                  NA                  NA                  NA                  NA                  NA                  NA       -10943.966206                   1        21889.933128        21896.560609                  17                  17                  17

loglik selects
model                                 loglik               diff
nagarch_skew_student_t           9467.949008           0.000000  best
gjr_skew_student_t               9476.328192           8.379183
egarch_skew_student_t            9486.414747          18.465738
nagarch_student_t                9491.081995          23.132987
st_student_t                     9491.623434          23.674426
gjr_student_t                    9499.237140          31.288131
egarch_student_t                 9509.352788          41.403780
garch_skew_student_t             9515.142722          47.193713
garch_student_t                  9534.244640          66.295632
nagarch_normal                   9553.505456          85.556447
st_normal                        9553.979401          86.030393
gjr_normal                       9561.629080          93.680072
igarch_student_t                 9576.571828         108.622820
egarch_normal                    9576.848928         108.899920
garch_normal                     9603.501097         135.552089
igarch_normal                    9658.535868         190.586859
constant_vol                    10943.966206        1476.017197

AICC selects
model                                   AICC               diff
nagarch_skew_student_t          18949.918092           0.000000  best
gjr_skew_student_t              18966.676459          16.758366
egarch_skew_student_t           18986.849569          36.931477
nagarch_student_t               18994.179044          44.260952
st_student_t                    18997.266944          47.348851
gjr_student_t                   19010.489334          60.571241
egarch_student_t                19030.720630          80.802538
garch_skew_student_t            19042.300497          92.382405
garch_student_t                 19078.500031         128.581939
nagarch_normal                  19117.021663         167.103570
st_normal                       19119.973856         170.055764
gjr_normal                      19133.268911         183.350819
igarch_student_t                19159.147955         209.229863
egarch_normal                   19163.708607         213.790515
garch_normal                    19215.009360         265.091268
igarch_normal                   19321.073884         371.155792
constant_vol                    21889.933128        2940.015035

BIC selects
model                                    BIC               diff
nagarch_skew_student_t          18996.295401           0.000000  best
gjr_skew_student_t              19013.053768          16.758366
egarch_skew_student_t           19033.226878          36.931477
nagarch_student_t               19033.933177          37.637776
st_student_t                    19043.644253          47.348851
gjr_student_t                   19050.243466          53.948065
egarch_student_t                19070.474763          74.179361
garch_skew_student_t            19082.054630          85.759228
garch_student_t                 19111.630269         115.334868
nagarch_normal                  19150.151900         153.856499
st_normal                       19159.727989         163.432588
gjr_normal                      19166.399149         170.103748
igarch_student_t                19179.028250         182.732848
egarch_normal                   19196.838845         200.543444
garch_normal                    19241.514985         245.219584
igarch_normal                   19334.328131         338.032730
constant_vol                    21896.560609        2900.265208

standardized residual stats
model                              n           mean             sd           skew    ex_kurtosis            min            max
nagarch_normal                  5587         0.0058         1.0004        -0.3588         1.5519        -8.3746         4.0092
nagarch_student_t               5587        -0.0125         1.0022        -0.3573         1.5859        -8.4535         4.0083
nagarch_skew_student_t          5587         0.0059         1.0011        -0.3585         1.5808        -8.4134         4.0323
garch_student_t                 5587        -0.0370         1.0008        -0.3625         1.5803        -8.4536         4.1061
garch_skew_student_t            5587        -0.0205         1.0040        -0.3641         1.5735        -8.4279         4.1518
garch_normal                    5587        -0.0232         0.9997        -0.3629         1.5668        -8.3760         4.1415
gjr_student_t                   5587        -0.0162         1.0019        -0.3634         1.6269        -8.6347         4.0674
gjr_skew_student_t              5587         0.0024         1.0012        -0.3639         1.6114        -8.5736         4.0896
gjr_normal                      5587         0.0028         1.0000        -0.3625         1.5628        -8.4641         4.0469
egarch_student_t                5587        -0.0168         1.0020        -0.3654         1.6737        -8.5965         4.2269
egarch_skew_student_t           5587         0.0019         1.0013        -0.3663         1.6668        -8.5566         4.2754
egarch_normal                   5587         0.0028         1.0001        -0.3652         1.6290        -8.4675         4.2166
igarch_student_t                5587        -0.0389         1.0506        -0.3936         1.5652        -8.4937         4.2541
igarch_normal                   5587        -0.0188         1.0490        -0.3913         1.5516        -8.3825         4.3211
st_student_t                    5587        -0.0124         1.0014        -0.3584         1.6039        -8.5060         4.0038
st_normal                       5587         0.0062         0.9996        -0.3592         1.5586        -8.3973         4.0081
constant_vol                    5587        -0.0000         1.0000         0.0245        15.9021       -10.2966        11.9363

conditional sd stats
model                              n           mean             sd           skew    ex_kurtosis            min            max
nagarch_normal                  5587         1.4581         0.8395         4.8174        32.2029         0.7096         9.7355
nagarch_student_t               5587         1.4579         0.8443         4.8023        32.1830         0.6933         9.8920
nagarch_skew_student_t          5587         1.4599         0.8483         4.8118        32.2566         0.6961         9.8779
garch_student_t                 5587         1.4669         0.8322         4.7031        30.5093         0.7362         9.4532
garch_skew_student_t            5587         1.4637         0.8333         4.6908        30.3410         0.7293         9.4232
garch_normal                    5587         1.4684         0.8312         4.7003        30.4385         0.7390         9.4137
gjr_student_t                   5587         1.4578         0.8248         4.4923        27.8543         0.7508         9.3773
gjr_skew_student_t              5587         1.4593         0.8285         4.4913        27.7944         0.7513         9.3615
gjr_normal                      5587         1.4579         0.8185         4.5077        27.9039         0.7573         9.1748
egarch_student_t                5587         1.4526         0.7412         3.8751        22.3325         0.6039         8.0387
egarch_skew_student_t           5587         1.4539         0.7436         3.8757        22.3286         0.6081         8.0443
egarch_normal                   5587         1.4518         0.7306         3.8790        22.3262         0.6242         7.8637
igarch_student_t                5587         1.4466         0.9260         4.2950        25.7689         0.5032         9.4467
igarch_normal                   5587         1.4482         0.9229         4.2854        25.6041         0.5114         9.3632
st_student_t                    5587         1.4588         0.8426         4.7660        31.6267         0.7009         9.8150
st_normal                       5587         1.4592         0.8386         4.7940        31.8423         0.7150         9.6911
constant_vol                    5587         1.7158         0.0000         1.0000        -2.0000         1.7158         1.7158

==== symbol: TLT ====
price observations: 5588, log returns used: 5587
         n           mean             sd           skew    ex_kurtosis            min            max
      5587         0.0133         0.9156        -0.0062         3.3784        -6.9010         7.2502

autocorrelations (lag 1-5)
   lag      returns    |returns|    returns^2
     1       -0.028        0.235        0.376
     2       -0.053        0.198        0.294
     3       -0.027        0.195        0.299
     4        0.005        0.216        0.319
     5       -0.004        0.189        0.242

model                               uncond_sd                  mu               omega               alpha                beta               gamma               shift                skew                 dof              loglik            n_params                AICC                 BIC         loglik_rank           AICC_rank            BIC_rank
nagarch_normal                       0.893613            0.004642            0.006112            0.051859            0.940007           -0.096126                  NA            0.000000                  NA        -6904.787907                   5        13819.586564        13852.716802                  14                  13                  13
nagarch_student_t                    0.892225            0.008497            0.004699            0.047793            0.945766           -0.106125                  NA            0.000000           15.296091        -6877.933251                   6        13767.881555        13807.635688                   8                   6                   7
nagarch_skew_student_t               0.891127            0.004358            0.004855            0.048262            0.945086           -0.105577                  NA           -0.060455           15.048638        -6873.243715                   7        13760.507506        13806.884815                   2                   3                   5
garch_student_t                      0.945870            0.006895            0.004858            0.049620            0.944951                  NA                  NA                  NA           14.923145        -6877.803703                   5        13765.618158        13798.748396                   7                   5                   2
garch_skew_student_t                 0.942497            0.002406            0.005063            0.050009            0.944291                  NA                  NA           -0.060138           14.666462        -6873.247793                   6        13758.510640        13798.264773                   3                   2                   1
garch_normal                         0.929588            0.002524            0.006332            0.053608            0.939065                  NA                  NA                  NA                  NA        -6904.536797                   4        13817.080760        13843.586385                  12                  12                  11
gjr_student_t                        0.981679            0.009768            0.004621            0.056289            0.945829           -0.013826                  NA                  NA           15.145960        -6876.286959                   6        13764.588972        13804.343105                   4                   4                   4
gjr_skew_student_t                   0.970054            0.005367            0.004813            0.056814            0.945095           -0.014048                  NA           -0.060263           14.937585        -6871.708392                   7        13757.436859        13803.814168                   1                   1                   3
gjr_normal                           0.951529            0.006213            0.006019            0.060179            0.940533           -0.014719                  NA                  NA                  NA        -6902.599386                   5        13815.209524        13848.339762                  11                  11                  12
egarch_student_t                           NA            0.010117           -0.001644            0.111124            0.991339                  NA                  NA                  NA           14.611212        -6882.631230                   6        13777.277514        13817.031647                   9                   9                  10
egarch_skew_student_t                      NA            0.005219           -0.001829            0.112244            0.991090                  NA                  NA           -0.062306           14.424947        -6877.733410                   7        13769.486895        13815.864204                   6                   8                   9
egarch_normal                              NA            0.006415           -0.002455            0.121439            0.988771                  NA                  NA                  NA                  NA        -6909.787106                   5        13829.584962        13862.715200                  15                  15                  15
igarch_student_t                           NA            0.006454                  NA            0.042302            0.957698                  NA                  NA            0.000000           14.796586        -6890.709017                   3        13787.422334        13807.302628                  10                  10                   6
igarch_normal                              NA            0.002256                  NA            0.042900            0.957100                  NA                  NA            0.000000                  NA        -6924.239805                   2        13852.481759        13865.736005                  16                  16                  16
st_student_t                         0.842569            0.009295            0.004490            0.047711            0.945955            0.000018           -0.139679            0.000000           15.241334        -6877.621360                   7        13769.262796        13815.640105                   5                   7                   8
st_normal                            0.860782            0.005334            0.005960            0.051695            0.940242            0.000039           -0.117576            0.000000                  NA        -6904.566474                   6        13821.148001        13860.902134                  13                  14                  14
constant_vol                         0.915554            0.000000            0.838239                  NA                  NA                  NA                  NA                  NA                  NA        -7434.690030                   1        14871.380776        14878.008258                  17                  17                  17

loglik selects
model                                 loglik               diff
gjr_skew_student_t               6871.708392           0.000000  best
nagarch_skew_student_t           6873.243715           1.535324
garch_skew_student_t             6873.247793           1.539401
gjr_student_t                    6876.286959           4.578567
st_student_t                     6877.621360           5.912968
egarch_skew_student_t            6877.733410           6.025018
garch_student_t                  6877.803703           6.095312
nagarch_student_t                6877.933251           6.224859
egarch_student_t                 6882.631230          10.922838
igarch_student_t                 6890.709017          19.000626
gjr_normal                       6902.599386          30.890994
garch_normal                     6904.536797          32.828405
st_normal                        6904.566474          32.858082
nagarch_normal                   6904.787907          33.079515
egarch_normal                    6909.787106          38.078714
igarch_normal                    6924.239805          52.531413
constant_vol                     7434.690030         562.981638

AICC selects
model                                   AICC               diff
gjr_skew_student_t              13757.436859           0.000000  best
garch_skew_student_t            13758.510640           1.073781
nagarch_skew_student_t          13760.507506           3.070647
gjr_student_t                   13764.588972           7.152113
garch_student_t                 13765.618158           8.181299
nagarch_student_t               13767.881555          10.444696
st_student_t                    13769.262796          11.825936
egarch_skew_student_t           13769.486895          12.050036
egarch_student_t                13777.277514          19.840655
igarch_student_t                13787.422334          29.985475
gjr_normal                      13815.209524          57.772664
garch_normal                    13817.080760          59.643901
nagarch_normal                  13819.586564          62.149705
st_normal                       13821.148001          63.711142
egarch_normal                   13829.584962          72.148103
igarch_normal                   13852.481759          95.044900
constant_vol                    14871.380776        1113.943917

BIC selects
model                                    BIC               diff
garch_skew_student_t            13798.264773           0.000000  best
garch_student_t                 13798.748396           0.483623
gjr_skew_student_t              13803.814168           5.549396
gjr_student_t                   13804.343105           6.078332
nagarch_skew_student_t          13806.884815           8.620043
igarch_student_t                13807.302628           9.037856
nagarch_student_t               13807.635688           9.370915
st_student_t                    13815.640105          17.375332
egarch_skew_student_t           13815.864204          17.599432
egarch_student_t                13817.031647          18.766875
garch_normal                    13843.586385          45.321613
gjr_normal                      13848.339762          50.074989
nagarch_normal                  13852.716802          54.452030
st_normal                       13860.902134          62.637361
egarch_normal                   13862.715200          64.450427
igarch_normal                   13865.736005          67.471233
constant_vol                    14878.008258        1079.743485

standardized residual stats
model                              n           mean             sd           skew    ex_kurtosis            min            max
nagarch_normal                  5587        -0.0028         1.0037        -0.1040         0.7556        -6.5738         4.4752
nagarch_student_t               5587        -0.0074         1.0061        -0.1014         0.7739        -6.6144         4.5477
nagarch_skew_student_t          5587        -0.0022         1.0056        -0.1021         0.7721        -6.6101         4.5357
garch_student_t                 5587        -0.0051         1.0009        -0.0939         0.7611        -6.4488         4.6234
garch_skew_student_t            5587         0.0005         1.0007        -0.0947         0.7587        -6.4460         4.6132
garch_normal                    5587         0.0001         0.9999        -0.0972         0.7433        -6.4290         4.5461
gjr_student_t                   5587        -0.0087         1.0009        -0.1013         0.7650        -6.5911         4.4418
gjr_skew_student_t              5587        -0.0032         1.0009        -0.1021         0.7630        -6.5906         4.4365
gjr_normal                      5587        -0.0045         0.9999        -0.1042         0.7485        -6.5665         4.4160
egarch_student_t                5587        -0.0097         1.0008        -0.1061         0.7534        -6.3940         4.3383
egarch_skew_student_t           5587        -0.0036         1.0008        -0.1069         0.7509        -6.3875         4.3437
egarch_normal                   5587        -0.0053         0.9999        -0.1091         0.7299        -6.3555         4.3246
igarch_student_t                5587        -0.0043         1.0263        -0.0883         0.8534        -6.6817         4.8141
igarch_normal                   5587         0.0012         1.0267        -0.0889         0.8537        -6.6924         4.7985
st_student_t                    5587        -0.0083         1.0052        -0.1032         0.7802        -6.6533         4.5049
st_normal                       5587        -0.0035         1.0033        -0.1051         0.7595        -6.5993         4.4509
constant_vol                    5587        -0.0000         1.0000        -0.0062         3.3784        -7.5521         7.9044

conditional sd stats
model                              n           mean             sd           skew    ex_kurtosis            min            max
nagarch_normal                  5587         0.8668         0.2858         2.3117        11.2024         0.4325         3.4357
nagarch_student_t               5587         0.8659         0.2881         2.1922         9.9914         0.4226         3.3481
nagarch_skew_student_t          5587         0.8662         0.2878         2.2085        10.1450         0.4244         3.3593
garch_student_t                 5587         0.8709         0.2908         2.1506         9.8006         0.4197         3.3695
garch_skew_student_t            5587         0.8709         0.2901         2.1686         9.9673         0.4221         3.3777
garch_normal                    5587         0.8705         0.2878         2.2744        11.0440         0.4303         3.4515
gjr_student_t                   5587         0.8711         0.2938         2.2034         9.9640         0.4210         3.3987
gjr_skew_student_t              5587         0.8708         0.2931         2.2223        10.1378         0.4231         3.4080
gjr_normal                      5587         0.8706         0.2904         2.3191        11.1114         0.4319         3.4702
egarch_student_t                5587         0.8689         0.2693         1.5513         5.3281         0.3746         3.0026
egarch_skew_student_t           5587         0.8687         0.2687         1.5635         5.4447         0.3759         3.0139
egarch_normal                   5587         0.8684         0.2655         1.6458         6.2752         0.3781         3.0996
igarch_student_t                5587         0.8611         0.3124         1.7644         6.5783         0.3367         3.2105
igarch_normal                   5587         0.8609         0.3129         1.7780         6.6912         0.3359         3.2280
st_student_t                    5587         0.8671         0.2900         2.1938         9.9404         0.4218         3.3590
st_normal                       5587         0.8674         0.2868         2.3119        11.1555         0.4320         3.4408
constant_vol                    5587         0.9156         0.0000         1.0000        -2.0000         0.9156         0.9156

==== model ranks by asset and information criterion ====
SPY AICC: nagarch_skew_student_t   egarch_skew_student_t    nagarch_student_t        gjr_skew_student_t       st_student_t             egarch_student_t         gjr_student_t            garch_skew_student_t     nagarch_normal           garch_student_t          st_normal                egarch_normal            igarch_student_t         gjr_normal               garch_normal             igarch_normal            constant_vol            
SPY  BIC: nagarch_skew_student_t   egarch_skew_student_t    nagarch_student_t        gjr_skew_student_t       st_student_t             egarch_student_t         gjr_student_t            garch_skew_student_t     nagarch_normal           garch_student_t          st_normal                igarch_student_t         egarch_normal            gjr_normal               garch_normal             igarch_normal            constant_vol            
EFA AICC: nagarch_skew_student_t   egarch_skew_student_t    gjr_skew_student_t       nagarch_student_t        egarch_student_t         st_student_t             gjr_student_t            garch_skew_student_t     garch_student_t          igarch_student_t         nagarch_normal           st_normal                egarch_normal            gjr_normal               garch_normal             igarch_normal            constant_vol            
EFA  BIC: nagarch_skew_student_t   egarch_skew_student_t    gjr_skew_student_t       nagarch_student_t        egarch_student_t         st_student_t             gjr_student_t            garch_skew_student_t     garch_student_t          igarch_student_t         nagarch_normal           egarch_normal            st_normal                gjr_normal               garch_normal             igarch_normal            constant_vol            
EEM AICC: nagarch_skew_student_t   gjr_skew_student_t       egarch_skew_student_t    nagarch_student_t        st_student_t             gjr_student_t            egarch_student_t         garch_skew_student_t     garch_student_t          nagarch_normal           st_normal                gjr_normal               igarch_student_t         egarch_normal            garch_normal             igarch_normal            constant_vol            
EEM  BIC: nagarch_skew_student_t   gjr_skew_student_t       egarch_skew_student_t    nagarch_student_t        st_student_t             gjr_student_t            egarch_student_t         garch_skew_student_t     garch_student_t          nagarch_normal           st_normal                gjr_normal               igarch_student_t         egarch_normal            garch_normal             igarch_normal            constant_vol            
TLT AICC: gjr_skew_student_t       garch_skew_student_t     nagarch_skew_student_t   gjr_student_t            garch_student_t          nagarch_student_t        st_student_t             egarch_skew_student_t    egarch_student_t         igarch_student_t         gjr_normal               garch_normal             nagarch_normal           st_normal                egarch_normal            igarch_normal            constant_vol            
TLT  BIC: garch_skew_student_t     garch_student_t          gjr_skew_student_t       gjr_student_t            nagarch_skew_student_t   igarch_student_t         nagarch_student_t        st_student_t             egarch_skew_student_t    egarch_student_t         garch_normal             gjr_normal               nagarch_normal           st_normal                egarch_normal            igarch_normal            constant_vol            

==== AICC rank counts ====
model                           AICC_avg         #1         #2         #3         #4         #5         #6         #7         #8         #9        #10        #11        #12        #13        #14        #15        #16        #17
nagarch_skew_student_t         15641.680          3          0          1          0          0          0          0          0          0          0          0          0          0          0          0          0          0
egarch_skew_student_t          15676.655          0          2          1          0          0          0          0          1          0          0          0          0          0          0          0          0          0
gjr_skew_student_t             15687.768          1          1          1          1          0          0          0          0          0          0          0          0          0          0          0          0          0
nagarch_student_t              15697.206          0          0          1          2          0          1          0          0          0          0          0          0          0          0          0          0          0
st_student_t                   15707.452          0          0          0          0          2          1          1          0          0          0          0          0          0          0          0          0          0
egarch_student_t               15727.031          0          0          0          0          1          1          1          0          1          0          0          0          0          0          0          0          0
gjr_student_t                  15734.901          0          0          0          1          0          1          2          0          0          0          0          0          0          0          0          0          0
garch_skew_student_t           15785.505          0          1          0          0          0          0          0          3          0          0          0          0          0          0          0          0          0
garch_student_t                15820.041          0          0          0          0          1          0          0          0          2          1          0          0          0          0          0          0          0
nagarch_normal                 15859.398          0          0          0          0          0          0          0          0          1          1          1          0          1          0          0          0          0
st_normal                      15870.313          0          0          0          0          0          0          0          0          0          0          2          1          0          1          0          0          0
igarch_student_t               15886.163          0          0          0          0          0          0          0          0          0          2          0          0          2          0          0          0          0
egarch_normal                  15904.469          0          0          0          0          0          0          0          0          0          0          0          1          1          1          1          0          0
gjr_normal                     15908.212          0          0          0          0          0          0          0          0          0          0          1          1          0          2          0          0          0
garch_normal                   16003.201          0          0          0          0          0          0          0          0          0          0          0          1          0          0          3          0          0
igarch_normal                  16134.538          0          0          0          0          0          0          0          0          0          0          0          0          0          0          0          4          0
constant_vol                   18349.037          0          0          0          0          0          0          0          0          0          0          0          0          0          0          0          0          4

==== BIC rank counts ====
model                            BIC_avg         #1         #2         #3         #4         #5         #6         #7         #8         #9        #10        #11        #12        #13        #14        #15        #16        #17
nagarch_skew_student_t         15688.057          3          0          0          0          1          0          0          0          0          0          0          0          0          0          0          0          0
egarch_skew_student_t          15723.032          0          2          1          0          0          0          0          0          1          0          0          0          0          0          0          0          0
gjr_skew_student_t             15734.145          0          1          2          1          0          0          0          0          0          0          0          0          0          0          0          0          0
nagarch_student_t              15736.960          0          0          1          2          0          0          1          0          0          0          0          0          0          0          0          0          0
st_student_t                   15753.829          0          0          0          0          2          1          0          1          0          0          0          0          0          0          0          0          0
egarch_student_t               15766.785          0          0          0          0          1          1          1          0          0          1          0          0          0          0          0          0          0
gjr_student_t                  15774.655          0          0          0          1          0          1          2          0          0          0          0          0          0          0          0          0          0
garch_skew_student_t           15825.259          1          0          0          0          0          0          0          3          0          0          0          0          0          0          0          0          0
garch_student_t                15853.171          0          1          0          0          0          0          0          0          2          1          0          0          0          0          0          0          0
nagarch_normal                 15892.528          0          0          0          0          0          0          0          0          1          1          1          0          1          0          0          0          0
igarch_student_t               15906.043          0          0          0          0          0          1          0          0          0          1          0          1          1          0          0          0          0
st_normal                      15910.067          0          0          0          0          0          0          0          0          0          0          2          0          1          1          0          0          0
egarch_normal                  15937.599          0          0          0          0          0          0          0          0          0          0          0          1          1          1          1          0          0
gjr_normal                     15941.342          0          0          0          0          0          0          0          0          0          0          0          2          0          2          0          0          0
garch_normal                   16029.707          0          0          0          0          0          0          0          0          0          0          1          0          0          0          3          0          0
igarch_normal                  16147.792          0          0          0          0          0          0          0          0          0          0          0          0          0          0          0          4          0
constant_vol                   18355.664          0          0          0          0          0          0          0          0          0          0          0          0          0          0          0          0          4

==== total fit time by model (seconds) ====
model                          seconds   n_params
nagarch_skew_student_t         304.402          7
st_student_t                    34.540          7
st_normal                       16.804          6
nagarch_student_t               10.157          6
nagarch_normal                   3.536          5
igarch_student_t                 2.624          3
igarch_normal                    0.828          2
egarch_skew_student_t            0.440          7
gjr_skew_student_t               0.419          7
garch_skew_student_t             0.359          6
egarch_student_t                 0.337          6
gjr_student_t                    0.270          6
garch_student_t                  0.197          5
egarch_normal                    0.184          5
gjr_normal                       0.165          5
garch_normal                     0.147          4
constant_vol                     0.000          1

conditional sd correlation matrices written to sd_corr.csv

volatility series written to:
  spy_vols.csv
  efa_vols.csv
  eem_vols.csv
  tlt_vols.csv

model parameter CSV written to model_params.csv

         assets         models/asset   total models   seconds        sec/model      
              4             17             68        376.631          5.539
```
