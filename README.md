# 근무 전 안전검사를 위한 스마트 키오스크 시스템

<div style="text-align: right"> 2023.12.13 발표 </div>

- 안전장구류 착용여부 비전 검사<br>-> [Yolact](https://github.com/dbolya/yolact)([설명](/Yolact_README.md))와 AI Hub의 [공사현장 안전장비 인식 이미지](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=163) 데이터셋 사용하여 구현
- 근무자의 신체상태 검사 - 심박수, 체온, 음주여부<br>신체상태는 Arduino와 센서를 사용하여 검사
- 검사 결과는 Firebase(Firestore Database 사용)에 저장
- Firebase에 저장된 데이터는 관리자용 어플을 통해 확인 가능
- [관리자용 어플](https://github.com/PoloCeleste/LifeVision_App)은 Flutter를 사용하여 구현

<br><br>
[1학기 시연 영상](https://www.youtube.com/watch?v=SbFfPqBx5S4) <br>
[2학기 시연 영상](https://www.youtube.com/watch?v=khaTaEOExCo)<br>
