action第一个数值是0，指代0号agent，这个是固定的（以后有多agent就不固定了）
第二个数值是action的类型
0. idleAction, 后面两个参数无效
1. attackAction, 后面两个参数是坐标，(范围是0-39, 1600)
2. collectAction, 后面两个参数是坐标，(范围是0-39, 1600)
3. pickupAction, 后面两个参数是类别，数量，类别是ismaterial的所有item
4. consumeAction, 后面两个参数是类别，数量
5. equipAction, 后面两个参数是类别，穿脱(0穿1脱)
6. synthesisAction, 合成，后面两个参数是类别，数量
7. throwAwayAction, 扔东西，后面两个参数是类别，数量
8. moveParallelAction, 时间片移动法, 后面两个参数是坐标，(范围是0-39, 1600)
详情参考src/jihuang/single/JiHuangGame.cpp:102起跳转查看。