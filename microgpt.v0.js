/*
 * MicroGPT Node.js Port (Final Optimized Version)
 * 포트: Prototype & Namespace architecture (객체 지향적 설계)
 * 기능: 자동 미분(Autograd), GPT 아키텍처, Adam 최적화, 모델 저장/로드 및 생성
 * -----------------------------------------------------------
 * [학습 가이드: v0 버전의 핵심]
 * 1. 동기식 학습: 데이터를 한꺼번에 메모리에 올려 순차적으로 처리하는 정석적인 방식입니다.
 * 2. Prototype 기반: Value 객체를 통해 연산 그래프를 빌드하는 과정을 깊이 있게 이해할 수 있습니다.
 * 3. Matrix 연산: 행렬 곱셈과 소프트맥스가 추상화되어 있어 신경망의 수학적 흐름이 잘 보입니다.
 */

var fs = require('fs');
var https = require('https');
var path = require('path');

// ==========================================
// 0. Namespace & Global Config
// 시스템 전체의 설정을 정의하고 모듈화된 네임스페이스를 구축합니다.
// ==========================================
var MicroGPT = {
	Config: {
		seed: 42, // 무작위성을 통제하여 언제 실행해도 같은 학습 결과를 얻게 합니다.
		inputPath: path.join(__dirname, 'input.txt'),
		modelPath: path.join(__dirname, 'model_weights.json'), // 학습 결과가 저장될 위치
		sourceUrl: 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
	},
	Utils: {},
	Autograd: {},
	Model: {},
	Core: {}
};

// ==========================================
// 1. Utils (Math & Helpers)
// 수학적 계산과 데이터 처리를 돕는 유틸리티입니다.
// ==========================================
MicroGPT.Utils = (function () {
	var seed = MicroGPT.Config.seed;
	// 시드 기반 난수: 인공지능이 '무작위성' 속에서도 일관성을 유지하게 합니다.
	function random() {
		var x = Math.sin(seed++) * 10000;
		return x - Math.floor(x);
	}
	// 가우시안 분포: 모델 가중치를 초기화할 때 0 주변의 값들로 골고루 분포시킵니다.
	function gauss(mu, sigma) {
		var u1 = 0, u2 = 0;
		while (u1 === 0) u1 = random();
		while (u2 === 0) u2 = random();
		var z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
		return z * sigma + mu;
	}
	// 데이터 셔플: 데이터 순서에 편향되지 않도록 학습 데이터를 섞어줍니다.
	function shuffle(array) {
		for (var i = array.length - 1; i > 0; i--) {
			var j = Math.floor(random() * (i + 1));
			var temp = array[i];
			array[i] = array[j];
			array[j] = temp;
		}
		return array;
	}
	// 가중치 기반 선택: 다음 글자를 고를 때 '확률'에 따라 뽑는 핵심 로직입니다.
	function choices(population, weights) {
		var total = weights.reduce(function (a, b) { return a + b; }, 0);
		var r = random() * total;
		var upTo = 0;
		for (var i = 0; i < population.length; i++) {
			upTo += weights[i];
			if (r <= upTo) return population[i];
		}
		return population[population.length - 1];
	}
	return { random: random, gauss: gauss, shuffle: shuffle, choices: choices };
})();

// ==========================================
// 2. Autograd Engine (자동 미분 엔진)
// 신경망 학습의 심장부입니다. 모든 연산의 '미분값'을 자동으로 추적합니다.
// ==========================================
MicroGPT.Autograd.Value = (function () {
	// Value 클래스: 숫자 데이터와 그 숫자가 만들어진 '역사(children)'를 저장합니다.
	function Value(data, children, local_grads) {
		this.data = data; // 현재 값
		this.grad = 0;    // 손실 함수에 대한 이 값의 민감도 (미분값)
		this._children = children || []; // 이 값을 만든 연산의 입력값들
		this._local_grads = local_grads || []; // 연산 당시의 로컬 미분값 (상수)
	}

	// 연산 오버로딩: 사칙연산 시 자동으로 '미분 규칙'을 그래프에 추가합니다.
	Value.prototype.add = function (other) {
		other = (other instanceof Value) ? other : new Value(other);
		return new Value(this.data + other.data, [this, other], [1, 1]);
	};
	// 곱셈 연산 및 미분 규칙 (상대방의 값)
	Value.prototype.mul = function (other) {
		other = (other instanceof Value) ? other : new Value(other);
		return new Value(this.data * other.data, [this, other], [other.data, this.data]);
	};
	// 지수 연산 (Power)
	Value.prototype.pow = function (exp) {
		return new Value(Math.pow(this.data, exp), [this], [exp * Math.pow(this.data, exp - 1)]);
	};
	// 로그 연산: 손실 함수(Loss) 계산 시 사용 (log 0 방지 epsilon 포함)
	Value.prototype.log = function () {
		return new Value(Math.log(this.data + 1e-10), [this], [1 / (this.data + 1e-10)]);
	};
	// 자연상수 e의 거듭제곱: Softmax 연산 시 사용
	Value.prototype.exp = function () {
		var res = Math.exp(this.data);
		return new Value(res, [this], [res]);
	};
	// ReLU 활성화 함수: 신경망에 비선형성을 부여 (0 이하는 차단)
	Value.prototype.relu = function () { // 비선형성 추가: 모델이 복잡한 패턴을 배우게 함
		return new Value(Math.max(0, this.data), [this], [(this.data > 0 ? 1 : 0)]);
	};
	Value.prototype.neg = function () { return this.mul(-1); };
	Value.prototype.sub = function (other) { return this.add(other instanceof Value ? other.neg() : new Value(-other)); };
	// 나눗셈 및 단항 음수화
	Value.prototype.div = function (other) { return this.mul(other instanceof Value ? other.pow(-1) : new Value(other).pow(-1)); };

	// 역전파(Backpropagation): 위상 정렬을 사용하여 연산의 끝에서부터 처음으로 미분값을 전파합니다.
	Value.prototype.backward = function () {
		var topo = [], visited = new Set();
		function build_topo(v) {
			if (!visited.has(v)) {
				visited.add(v);
				v._children.forEach(build_topo);
				topo.push(v); // 모든 부모 노드가 처리된 후 리스트에 추가 (위상 정렬)
			}
		}
		build_topo(this);
		this.grad = 1; // 최종 출력(Loss)의 자기 자신에 대한 미분은 1
		for (var i = topo.length - 1; i >= 0; i--) {
			var v = topo[i];
			for (var j = 0; j < v._children.length; j++) {
				// 연쇄 법칙(Chain Rule): (전체 미분) = (로컬 미분) * (상위 미분)
				v._children[j].grad += v._local_grads[j] * v.grad;
			}
		}
	};
	return Value;
})();

// ==========================================
// 3. GPT Model Architecture
// GPT-2/3 스타일의 트랜스포머 아키텍처를 구현합니다.
// ==========================================
MicroGPT.Model = (function () {
	var Value = MicroGPT.Autograd.Value;
	var Utils = MicroGPT.Utils;

	// 모델의 초기 지능(가중치 행렬)을 생성합니다.
	function createMatrix(rows, cols, std) {
		var mat = [];
		for (var i = 0; i < rows; i++) {
			var row = [];
			for (var j = 0; j < cols; j++) row.push(new Value(Utils.gauss(0, std || 0.02)));
			mat.push(row);
		}
		return mat;
	}

	// 선형 레이어: 입력 벡터에 가중치를 곱해 새로운 특징을 추출합니다.
	function linear(x, w) {
		return w.map(function (row) {
			var sum = new Value(0);
			for (var i = 0; i < row.length; i++) sum = sum.add(row[i].mul(x[i]));
			return sum;
		});
	}

	// 소프트맥스: 출력값들을 합이 1인 확률값으로 변환하여 '정답 확률'을 계산합니다.
	function softmax(logits) {
		var max_val = logits.reduce(function (m, v) { return Math.max(m, v.data); }, -Infinity);
		var exps = logits.map(function (v) { return v.sub(max_val).exp(); });
		var sum_exps = exps.reduce(function (a, b) { return a.add(b); }, new Value(0));
		return exps.map(function (e) { return e.div(sum_exps); });
	}

	// 정규화: 학습 중 값이 너무 커지거나 작아지지 않게 표준화하여 안정성을 높입니다.
	function rmsnorm(x) {
		var ss = x.reduce(function (a, b) { return a.add(b.mul(b)); }, new Value(0)).div(x.length);
		var inv_std = ss.add(1e-5).pow(-0.5);
		return x.map(function (v) { return v.mul(inv_std); });
	}

	// GPT 순전파(Forward Pass): 토큰 입력부터 다음 토큰 예측까지의 전체 연산 과정
	function forward(token_id, pos_id, keys, values, state_dict, config) {
		// 1. 가중치 조회(Embedding): 글자 ID와 위치 ID를 결합하여 의미 있는 벡터 생성
		var x = state_dict.wte[token_id].map(function (v, i) { return v.add(state_dict.wpe[pos_id][i]); });
		x = rmsnorm(x);

		for (var l = 0; l < config.n_layer; l++) {
			var x_res = x; // 잔차 연결(Residual Connection)을 위해 원본 저장
			x = rmsnorm(x);

			// 2. Self-Attention: 현재 단어와 과거 단어들 사이의 관계 점수를 계산
			var q = linear(x, state_dict['l' + l + '.wq']);
			var k = linear(x, state_dict['l' + l + '.wk']);
			var v = linear(x, state_dict['l' + l + '.wv']);
			keys[l].push(k); values[l].push(v); // 과거 정보를 기억(KV Cache 방식)

			var heads = [];
			for (var h = 0; h < config.n_head; h++) {
				var start = h * config.head_dim, end = start + config.head_dim;
				var qh = q.slice(start, end);
				var kh_hist = keys[l].map(function (ki) { return ki.slice(start, end); });
				var vh_hist = values[l].map(function (vi) { return vi.slice(start, end); });

				// 닷-프로덕트 어텐션: 질문(Q)과 저장된 기억(K)을 비교하여 중요도 계산
				var scores = kh_hist.map(function (kh) {
					var dot = new Value(0);
					for (var i = 0; i < qh.length; i++) dot = dot.add(qh[i].mul(kh[i]));
					return dot.div(Math.sqrt(config.head_dim));
				});
				var att = softmax(scores);
				// 가중치 합: 계산된 중요도(att)만큼 실제 의미 정보(V)를 가져옴
				var hout = Array(config.head_dim).fill(0).map(function (_, i) {
					var sum = new Value(0);
					att.forEach(function (a, t) { sum = sum.add(a.mul(vh_hist[t][i])); });
					return sum;
				});
				heads = heads.concat(hout);
			}
			x = linear(heads, state_dict['l' + l + '.wo']).map(function (v, i) { return v.add(x_res[i]); });

			// 3. MLP (Feed Forward): 추출된 문맥 정보를 바탕으로 더 고차원적인 추론 수행
			x_res = x;
			x = rmsnorm(x);
			x = linear(x, state_dict['l' + l + '.w1']).map(function (v) { return v.relu(); });
			x = linear(x, state_dict['l' + l + '.w2']).map(function (v, i) { return v.add(x_res[i]); });
		}
		// 4. 언어 모델 헤드: 최종 벡터를 각 글자별 확률로 변환
		return linear(x, state_dict.lm_head);
	}

	return { createMatrix: createMatrix, forward: forward, softmax: softmax };
})();

// ==========================================
// 4. Core Logic (Train, Save, Load, Generate)
// ==========================================
MicroGPT.Core = {
	// 모델 저장: 모든 지식(가중치)을 숫자로 추출하여 JSON 파일로 저장합니다.
	saveModel: function (state_dict, uchars, config, filePath) {
		var data = { config: config, uchars: uchars, weights: {} };
		for (var k in state_dict) {
			data.weights[k] = state_dict[k].map(function (row) {
				return row.map(function (v) { return v.data; });
			});
		}
		fs.writeFileSync(filePath || MicroGPT.Config.modelPath, JSON.stringify(data));
		console.log("\n[Success] Model saved to: " + (filePath || MicroGPT.Config.modelPath));
	},

	// 모델 로드: 저장된 JSON 파일을 읽어 다시 Value 객체 그래프로 복원합니다.
	loadModel: function (filePath) {
		var p = filePath || MicroGPT.Config.modelPath;
		if (!fs.existsSync(p)) return null;
		var data = JSON.parse(fs.readFileSync(p, 'utf8'));
		var state_dict = {};
		for (var k in data.weights) {
			state_dict[k] = data.weights[k].map(function (row) {
				return row.map(function (d) { return new MicroGPT.Autograd.Value(d); });
			});
		}
		console.log("[Success] Model loaded.");
		return { state_dict: state_dict, uchars: data.uchars, config: data.config, BOS: data.uchars.length };
	},

	// 텍스트 생성: 학습된 지식을 사용하여 한 글자씩 '추론'하여 문장을 만듭니다.
	generate: function (count, modelData, temperature) {
		var temp = temperature || 0.8;
		var results = [];
		var sd = modelData.state_dict, cfg = modelData.config, uchars = modelData.uchars, BOS = modelData.BOS;

		for (var i = 0; i < count; i++) {
			var keys = [], values = [], sample = [], token_id = BOS;
			for (var l = 0; l < cfg.n_layer; l++) { keys.push([]); values.push([]); }

			for (var pos = 0; pos < cfg.block_size; pos++) {
				var logits = MicroGPT.Model.forward(token_id, pos, keys, values, sd, cfg);
				// 확률 조절: temperature가 낮으면 안전한 선택, 높으면 창의적인 선택을 합니다.
				var probs = MicroGPT.Model.softmax(logits.map(function (l) { return l.div(temp); }));
				var next = MicroGPT.Utils.choices([...Array(probs.length).keys()], probs.map(p => p.data));
				if (next === BOS) break;
				sample.push(uchars[next]);
				token_id = next;
			}
			results.push(sample.join(''));
		}
		return results;
	},

	// 학습 프로세스: 데이터를 읽어 모델을 '훈련'시킵니다. (동기식 방식)
	train: function (numSteps, callback) {
		var conf = MicroGPT.Config;
		// 1. 데이터 준비: 파일이 없으면 다운로드합니다.
		if (!fs.existsSync(conf.inputPath)) {
			console.log("Downloading dataset...");
			var file = fs.createWriteStream(conf.inputPath);
			https.get(conf.sourceUrl, function (res) {
				res.pipe(file).on('finish', function () { file.close(); MicroGPT.Core.train(numSteps, callback); });
			});
			return;
		}

		// 2. 어휘집 구축: 텍스트 파일 전체를 읽어 사용된 고유 글자들을 파악합니다.
		var text = fs.readFileSync(conf.inputPath, 'utf8');
		var docs = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
		var uchars = Array.from(new Set(docs.join(''))).sort();
		var BOS = uchars.length, vocab_size = BOS + 1;

		// 모델 하이퍼파라미터 설정
		var config = { n_layer: 1, n_embd: 16, block_size: 16, n_head: 4 };
		config.head_dim = config.n_embd / config.n_head;

		// 3. 모델 초기화: 가중치 행렬들을 생성합니다.
		var sd = {
			wte: MicroGPT.Model.createMatrix(vocab_size, config.n_embd),
			wpe: MicroGPT.Model.createMatrix(config.block_size, config.n_embd),
			lm_head: MicroGPT.Model.createMatrix(vocab_size, config.n_embd)
		};
		for (var i = 0; i < config.n_layer; i++) {
			sd['l' + i + '.wq'] = MicroGPT.Model.createMatrix(config.n_embd, config.n_embd);
			sd['l' + i + '.wk'] = MicroGPT.Model.createMatrix(config.n_embd, config.n_embd);
			sd['l' + i + '.wv'] = MicroGPT.Model.createMatrix(config.n_embd, config.n_embd);
			sd['l' + i + '.wo'] = MicroGPT.Model.createMatrix(config.n_embd, config.n_embd);
			sd['l' + i + '.w1'] = MicroGPT.Model.createMatrix(config.n_embd * 4, config.n_embd);
			sd['l' + i + '.w2'] = MicroGPT.Model.createMatrix(config.n_embd, config.n_embd * 4);
		}

		// 모든 가중치를 하나의 리스트로 모아 업데이트 준비를 합니다.
		var params = [];
		for (var k in sd) sd[k].forEach(row => row.forEach(p => params.push(p)));

		// Adam 최적화 파라미터 초기화
		var m = Array(params.length).fill(0), v = Array(params.length).fill(0);
		console.log("Training (" + numSteps + " steps)...");

		// 4. 학습 루프: 지정된 단계만큼 반복하여 똑똑해집니다.
		for (var s = 0; s < numSteps; s++) {
			var doc = docs[s % docs.length];
			// 텍스트를 숫자로 변환 (시작 토큰 + 본문 + 종료 토큰)
			var tokens = [BOS].concat(doc.split('').map(c => uchars.indexOf(c))).concat([BOS]);
			var n = Math.min(config.block_size, tokens.length - 1);
			var keys = [], values = [], losses = [];
			for (var l = 0; l < config.n_layer; l++) { keys.push([]); values.push([]); }

			// Forward pass: 예측 수행 및 손실(오차) 계산
			for (var p = 0; p < n; p++) {
				var logits = MicroGPT.Model.forward(tokens[p], p, keys, values, sd, config);
				var probs = MicroGPT.Model.softmax(logits);
				// 정답일 확률이 높을수록 손실값이 작아집니다.
				losses.push(probs[tokens[p + 1]].log().neg());
			}
			var loss = losses.reduce((a, b) => a.add(b), new MicroGPT.Autograd.Value(0)).div(n);

			// Backward pass: 오차 역전파 실행 (기울기 계산)
			params.forEach(p => p.grad = 0);
			loss.backward();

			// Adam Optimizer 적용: 계산된 기울기(grad)를 바탕으로 지능(data)을 업데이트
			var lr = 0.01 * (1 - s / numSteps); // 학습이 끝날수록 학습률을 점차 낮춤
			for (var i = 0; i < params.length; i++) {
				m[i] = 0.9 * m[i] + 0.1 * params[i].grad;
				v[i] = 0.99 * v[i] + 0.01 * Math.pow(params[i].grad, 2);
				var m_h = m[i] / (1 - Math.pow(0.9, s + 1));
				var v_h = v[i] / (1 - Math.pow(0.99, s + 1));
				params[i].data -= lr * m_h / (Math.sqrt(v_h) + 1e-8);
			}
			if (s % 10 === 0) process.stdout.write("Step " + s + " | Loss: " + loss.data.toFixed(4) + "\r");
		}
		console.log("\nDone.");
		if (callback) callback(sd, uchars, config, BOS);
	}
};

// ==========================================
// 5. Usage Example (사용법)
// 실제 프로그램을 구동하는 제어 영역입니다.
// ==========================================

// 실행 로직: 저장된 모델이 있으면 불러오고, 없으면 새로 학습합니다.
var savedData = MicroGPT.Core.loadModel();
if (savedData) {
	console.log("학습된 모델로 이름을 생성합니다:");
	var names = MicroGPT.Core.generate(10, savedData, 0.7);
	names.forEach((n, i) => console.log((i + 1) + ". " + n));
} else {
	console.log("저장된 모델이 없습니다. 먼저 학습(train)을 진행해주세요.");
	// 처음 실행 시 200단계 동안 학습을 진행하고 결과를 파일로 저장합니다.
	MicroGPT.Core.train(200, function (sd, uchars, config, BOS) {
		MicroGPT.Core.saveModel(sd, uchars, config);
		console.log("학습 완료 및 모델 저장됨. 다시 실행하면 로드하여 생성합니다.");
	});
}