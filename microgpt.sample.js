// 1. 엔진 로직 로드
require('./microgpt.js');

function run() {
	var model = MicroGPT.Core.load();
	if (model) {
		var names = MicroGPT.Core.generate(10, model, 0.7);
		console.log("생성된 결과:", names);
	} else {
		console.log("모델 파일이 없습니다. train.js를 먼저 실행하세요.");
	}
}
run();