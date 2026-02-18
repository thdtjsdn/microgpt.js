// 1. 엔진 로직 로드 (전역 변수 MicroGPT가 메모리에 생성됨)
require('./microgpt.js');

async function run() {
	console.log("학습을 시작합니다...");
	// 비동기 스트리밍 학습 호출
	await MicroGPT.Core.trainAsync(500, function(sd, uchars, config, BOS) {
		MicroGPT.Core.save(sd, uchars, config);
		console.log("학습 완료!");
	});
}
run();