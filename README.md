# 근무 전 안전검사를 위한 스마트 키오스크 시스템

<div style="text-align: right"> 2023.11.07 발표 <br> 2023.12.13 최종수정 </div>

- 안전장구류 착용여부 비전 검사<br>-> [Yolact](https://github.com/dbolya/yolact)([설명](/Yolact_README.md))와 AI Hub의 [공사현장 안전장비 인식 이미지](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=163) 데이터셋 사용하여 구현
- 비전 검사하여 착용되지 않은 [장구류의 착용법 안내](https://github.com/PoloCeleste/LifeVision_SafetyGearInfo)
- 근무자의 신체상태 검사 - 심박수, 체온, 음주여부<br>신체상태는 [Arduino](https://github.com/PoloCeleste/LifeVision_Arduio)와 센서를 사용하여 검사
- 검사 결과는 Firebase(Firestore Database 사용)에 저장
- Firebase에 저장된 데이터는 관리자용 어플을 통해 확인 가능
- [관리자용 어플](https://github.com/PoloCeleste/LifeVision_App)은 Flutter를 사용하여 구현
- 기존에는 Ubuntu(Jetson AGX Orin)에서 동작하는 것으로 제작하였으나 단독 실행은 잘 되는데 Pyqt5로 구현한 GUI창을 통하여 장구류 비전검사 실행 시 카메라에 접근하지 못하는 문제 발생하여 Windows로 이식.

<br><br>

- 구동시 필수
  - Python 3.9 ~ (구현 시 3.9.6과 3.11.8 사용)
  - JDK 19 (or 17)
  - Java로 제작된 안전장구류 착용법은 [Font](/JavaFont) 설치 요망
  - Nvidia CUDA, Pytorch 설치
  - pip패키지 ; numpy, OpenCV/Contrib, pyserial, matplotlib, schedule, pygame, cython, pillow, GitPython, termcolor, tensorboard, PyQt5, firebase-admin 등
  - 웹캠, 센서부(시리얼 포트 확인 필요), Firebase DB 연동
  - Nvidia GPU

<br><br>
[1학기 시연 영상](https://www.youtube.com/watch?v=SbFfPqBx5S4) <br>
[2학기 시연 영상](https://www.youtube.com/watch?v=khaTaEOExCo)<br>
