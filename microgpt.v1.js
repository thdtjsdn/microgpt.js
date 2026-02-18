/*
 * MicroGPT Node.js - High Performance & Full Integration
 * 기능: Autograd, GPT Architecture, Adam Optimizer, Save/Load, Generator
 * -----------------------------------------------------------
 * [학습 가이드: v1 최적화 버전의 특징]
 * 1. 성능 최적화: Array 생성 시 크기를 미리 지정하여 메모리 재할당 비용을 줄였습니다.
 * 2. 타입 배열 활용: Adam Optimizer에서 Float64Array를 사용하여 수치 계산 속도를 높였습니다.
 * 3. 코드 정제: Autograd 엔진의 불필요한 참조를 줄여 가비지 컬렉션 부담을 완화했습니다.
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
// 1. Utils (Performance Optimized)
// 수학 연산 및 샘플링을 위한 유틸리티 함수 모음입니다.
// ==========================================
MicroGPT.Utils = (function () {
	var seed = MicroGPT.Config.seed;
	return {
		// 결정론적 난수 생성: 동일한 시드에서 항상 같은 결과를 얻어 디버깅을 용이하게 합니다.
		random: function () {
			var x = Math.sin(seed++) * 10000;
			return x - Math.floor(x);
		},
		// 가우시안 정규 분포: 가중치 초기화 시 신경망의 초기 출력이 너무 튀지 않게 조절합니다.
		gauss: function (mu, sigma) {
			var u1 = 0; while (u1 === 0) u1 = this.random();
			var u2 = 0; while (u2 === 0) u2 = this.random();
			var z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
			return z * sigma + mu;
		},
		// 가중치 기반 확률 선택: 소프트맥스 결과(확률)에 따라 다음 글자를 무작위로 선택합니다.
		choices: function (population, weights) {
			var total = 0;
			for (var i = 0; i < weights.length; i++) total += weights[i];
			var r = this.random() * total;
			var upTo = 0;
			for (var i = 0; i < population.length; i++) {
				upTo += weights[i];
				if (r <= upTo) return population[i];
			}
			return population[population.length - 1];
		}
	};
})();

// ==========================================
// 2. Autograd Engine (Value Object)
// 역전파(Backpropagation)를 수행하는 자동 미분 엔진의 핵심입니다.
// ==========================================
MicroGPT.Autograd = (function () {
	// Value 객체: 값(data)과 기울기(grad), 그리고 연산 계보(children)를 저장합니다.
	function Value(data, children, local_grads) {
		this.data = data;
		this.grad = 0;
		this._children = children || null;    // 현재 노드를 만든 이전 노드들
		this._local_grads = local_grads || null; // 각 자식 노드에 대한 국소 미분값
	}

	// 덧셈 연산 및 미분 규칙 (1, 1)
	Value.prototype.add = function (other) {
		var o = (other instanceof Value) ? other : new Value(other);
		return new Value(this.data + o.data, [this, o], [1, 1]);
	};
	// 곱셈 연산 및 미분 규칙 (상대방의 값)
	Value.prototype.mul = function (other) {
		var o = (other instanceof Value) ? other : new Value(other);
		return new Value(this.data * o.data, [this, o], [o.data, this.data]);
	};
	// 지수 연산 (Power)
	Value.prototype.pow = function (exp) {
		return new Value(Math.pow(this.data, exp), [this], [exp * Math.pow(this.data, exp - 1)]);
	};
	// 로그 연산: 손실 함수(Loss) 계산 시 사용 (log 0 방지 epsilon 포함)
	Value.prototype.log = function () {
		var val = this.data + 1e-10;
		return new Value(Math.log(val), [this], [1 / val]);
	};
	// 자연상수 e의 거듭제곱: Softmax 연산 시 사용
	Value.prototype.exp = function () {
		var res = Math.exp(this.data);
		return new Value(res, [this], [res]);
	};
	// ReLU 활성화 함수: 신경망에 비선형성을 부여 (0 이하는 차단)
	Value.prototype.relu = function () {
		return new Value(this.data > 0 ? this.data : 0, [this], [this.data > 0 ? 1 : 0]);
	};
	// 나눗셈 및 단항 음수화
	Value.prototype.div = function (other) {
		return this.mul((other instanceof Value ? other : new Value(other)).pow(-1));
	};
	Value.prototype.neg = function () { return this.mul(-1); };

	// 역전파: 출력 노드부터 입력 노드까지 체인 룰(Chain Rule)을 적용하여 grad를 계산합니다.
	Value.prototype.backward = function () {
		var topo = [], visited = new Set();
		// 위상 정렬을 통해 연산 순서의 역순으로 리스트 생성
		function build(v) {
			if (visited.has(v)) return;
			visited.add(v);
			if (v._children) {
				for (var i = 0; i < v._children.length; i++) build(v._children[i]);
			}
			topo.push(v);
		}
		build(this);
		this.grad = 1; // 최종 출력의 자기 자신에 대한 미분은 1
		for (var i = topo.length - 1; i >= 0; i--) {
			var v = topo[i];
			if (!v._children) continue;
			for (var j = 0; j < v._children.length; j++) {
				// 부모의 grad와 자신의 로컬 미분값을 곱해 자식에게 전달
				v._children[j].grad += v._local_grads[j] * v.grad;
			}
		}
	};
	return { Value: Value };
})();

// ==========================================
// 3. GPT Model Architecture (Optimized For-loops)
// 트랜스포머 아키텍처의 핵심 연산을 정의합니다.
// ==========================================
MicroGPT.Model = (function () {
	var Value = MicroGPT.Autograd.Value;
	var Utils = MicroGPT.Utils;

	return {
		// 가중치 행렬 초기화
		createMatrix: function (rows, cols, std) {
			var mat = new Array(rows);
			for (var i = 0; i < rows; i++) {
				mat[i] = new Array(cols);
				for (var j = 0; j < cols; j++) mat[i][j] = new Value(Utils.gauss(0, std || 0.02));
			}
			return mat;
		},
		// 선형 레이어 (Matrix Multiplication): 가중치와 입력의 내적 합 계산
		linear: function (x, w) {
			var out = new Array(w.length);
			for (var i = 0; i < w.length; i++) {
				var row = w[i];
				var sum = new Value(0);
				for (var j = 0; j < row.length; j++) sum = sum.add(row[j].mul(x[j]));
				out[i] = sum;
			}
			return out;
		},
		// 소프트맥스: 출력 값을 0~1 사이의 확률값으로 변환하고 합이 1이 되게 정규화
		softmax: function (logits) {
			var max_val = -Infinity;
			// 수치 안정성을 위해 최대값을 뺌 (Overflow 방지)
			for (var i = 0; i < logits.length; i++) if (logits[i].data > max_val) max_val = logits[i].data;
			var exps = new Array(logits.length);
			var sum_exps = new Value(0);
			for (var i = 0; i < logits.length; i++) {
				exps[i] = logits[i].add(-max_val).exp();
				sum_exps = sum_exps.add(exps[i]);
			}
			for (var i = 0; i < exps.length; i++) exps[i] = exps[i].div(sum_exps);
			return exps;
		},
		// RMS 정규화: 학습 중 레이어 사이의 데이터 분포를 고르게 유지하여 안정성을 확보
		rmsnorm: function (x) {
			var ss = new Value(0);
			for (var i = 0; i < x.length; i++) ss = ss.add(x[i].mul(x[i]));
			var inv_std = ss.div(x.length).add(1e-5).pow(-0.5);
			var out = new Array(x.length);
			for (var i = 0; i < x.length; i++) out[i] = x[i].mul(inv_std);
			return out;
		},
		// Forward Pass: 입력 토큰으로부터 확률 분포까지의 순전파 경로
		forward: function (token_id, pos_id, keys, values, state_dict, config) {
			// 임베딩 테이블 참조 (글자 정보 + 위치 정보)
			var wte_row = state_dict.wte[token_id];
			var wpe_row = state_dict.wpe[pos_id];
			var x = new Array(config.n_embd);
			for (var i = 0; i < config.n_embd; i++) x[i] = wte_row[i].add(wpe_row[i]);
			x = this.rmsnorm(x);

			// 트랜스포머 레이어 반복
			for (var l = 0; l < config.n_layer; l++) {
				var x_attn_res = x; // Residual Connection (잔차 연결) 저장
				x = this.rmsnorm(x);
				// QKV 프로젝션 (Self-Attention의 기초)
				var q = this.linear(x, state_dict['l' + l + '.wq']);
				var k = this.linear(x, state_dict['l' + l + '.wk']);
				var v = this.linear(x, state_dict['l' + l + '.wv']);
				keys[l].push(k); values[l].push(v); // KV Cache에 저장

				// 멀티헤드 어텐션
				var heads_out = [];
				for (var h = 0; h < config.n_head; h++) {
					var start = h * config.head_dim;
					var scores = new Array(keys[l].length);
					for (var t = 0; t < keys[l].length; t++) {
						var dot = new Value(0);
						for (var d = 0; d < config.head_dim; d++) dot = dot.add(q[start + d].mul(keys[l][t][start + d]));
						scores[t] = dot.div(Math.sqrt(config.head_dim));
					}
					var att = this.softmax(scores);
					// 가중 합계 (Attention Value 계산)
					for (var d = 0; d < config.head_dim; d++) {
						var sum = new Value(0);
						for (var t = 0; t < att.length; t++) sum = sum.add(att[t].mul(values[l][t][start + d]));
						heads_out.push(sum);
					}
				}
				// 어텐션 결과 결합 및 잔차 합산
				x = this.linear(heads_out, state_dict['l' + l + '.wo']);
				for (var i = 0; i < x.length; i++) x[i] = x[i].add(x_attn_res[i]);

				// MLP (Feed Forward Network) 섹션
				var x_mlp_res = x;
				x = this.rmsnorm(x);
				x = this.linear(x, state_dict['l' + l + '.w1']);
				for (var i = 0; i < x.length; i++) x[i] = x[i].relu();
				x = this.linear(x, state_dict['l' + l + '.w2']);
				for (var i = 0; i < x.length; i++) x[i] = x[i].add(x_mlp_res[i]);
			}
			// 최종 헤드: 은닉 상태를 어휘(vocab) 크기의 확률로 변환
			return this.linear(x, state_dict.lm_head);
		}
	};
})();

// ==========================================
// 4. Core Logic (Save, Load, Train, Generate)
// 엔진의 핵심 동작(학습/생성/저장)을 제어합니다.
// ==========================================
MicroGPT.Core = {
	// 모델 가중치를 JSON으로 변환하여 파일로 저장
	save: function (state_dict, uchars, config) {
		var data = { config: config, uchars: uchars, weights: {} };
		for (var k in state_dict) {
			var mat = state_dict[k];
			var rawMat = new Array(mat.length);
			for (var r = 0; r < mat.length; r++) {
				rawMat[r] = new Array(mat[r].length);
				for (var c = 0; c < mat[r].length; c++) rawMat[r][c] = mat[r][c].data;
			}
			data.weights[k] = rawMat;
		}
		fs.writeFileSync(MicroGPT.Config.modelPath, JSON.stringify(data));
		console.log("\n[Save] Model stored at: " + MicroGPT.Config.modelPath);
	},

	// 저장된 파일에서 모델 데이터를 읽어 Value 객체로 복구
	load: function () {
		if (!fs.existsSync(MicroGPT.Config.modelPath)) return null;
		var data = JSON.parse(fs.readFileSync(MicroGPT.Config.modelPath, 'utf8'));
		var sd = {};
		for (var k in data.weights) {
			var rawMat = data.weights[k];
			sd[k] = new Array(rawMat.length);
			for (var r = 0; r < rawMat.length; r++) {
				sd[k][r] = new Array(rawMat[r].length);
				for (var c = 0; c < rawMat[r].length; c++) sd[k][r][c] = new MicroGPT.Autograd.Value(rawMat[r][c]);
			}
		}
		return { state_dict: sd, uchars: data.uchars, config: data.config, BOS: data.uchars.length };
	},

	// 텍스트 생성: 학습된 지식을 바탕으로 새로운 문장을 조립
	generate: function (count, modelData, temperature) {
		var temp = temperature || 0.8;
		var results = [];
		var sd = modelData.state_dict, cfg = modelData.config, uchars = modelData.uchars, BOS = modelData.BOS;

		for (var i = 0; i < count; i++) {
			var keys = [], values = [], sample = [], token_id = BOS;
			for (var l = 0; l < cfg.n_layer; l++) { keys.push([]); values.push([]); }
			for (var pos = 0; pos < cfg.block_size; pos++) {
				var logits = MicroGPT.Model.forward(token_id, pos, keys, values, sd, cfg);
				var scaledLogits = new Array(logits.length);
				// 온도(Temperature) 조절: 낮으면 보수적(정확), 높으면 창의적(다양) 결과가 나옴
				for (var j = 0; j < logits.length; j++) scaledLogits[j] = logits[j].div(temp);
				var probs = MicroGPT.Model.softmax(scaledLogits);
				var probData = new Array(probs.length);
				for (var j = 0; j < probs.length; j++) probData[j] = probs[j].data;
				var next = MicroGPT.Utils.choices([...Array(probs.length).keys()], probData);
				if (next === BOS) break; // 문장 종료 토큰 시 중단
				sample.push(uchars[next]);
				token_id = next;
			}
			results.push(sample.join(''));
		}
		return results;
	},

	// 학습 실행: 오차 역전파와 Adam Optimizer를 사용하여 모델을 훈련
	train: function (numSteps, callback) {
		var conf = MicroGPT.Config;
		// 데이터 부재 시 다운로드 시도
		if (!fs.existsSync(conf.inputPath)) {
			console.log("Downloading dataset...");
			var file = fs.createWriteStream(conf.inputPath);
			https.get(conf.sourceUrl, function (res) {
				res.pipe(file).on('finish', function () { file.close(); MicroGPT.Core.train(numSteps, callback); });
			});
			return;
		}

		// 데이터 전처리 및 어휘집 생성
		var text = fs.readFileSync(conf.inputPath, 'utf8');
		var docs = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
		var uchars = Array.from(new Set(docs.join(''))).sort();
		var BOS = uchars.length, vocab_size = BOS + 1; // 문장 시작/끝 토큰 포함

		var config = { n_layer: 1, n_embd: 16, block_size: 16, n_head: 4 };
		config.head_dim = config.n_embd / config.n_head;

		// 모델 초기화
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

		// 파라미터 리스트화 및 Adam 상태 배열 초기화 (성능을 위해 Float64Array 사용)
		var params = [];
		for (var k in sd) {
			for (var r = 0; r < sd[k].length; r++) {
				for (var c = 0; c < sd[k][r].length; c++) params.push(sd[k][r][c]);
			}
		}

		var m = new Float64Array(params.length), v = new Float64Array(params.length);
		console.log("Training Start...");

		// 메인 학습 루프
		for (var s = 0; s < numSteps; s++) {
			var doc = docs[Math.floor(MicroGPT.Utils.random() * docs.length)];
			var tokens = [BOS];
			for (var i = 0; i < doc.length; i++) tokens.push(uchars.indexOf(doc[i]));
			tokens.push(BOS);

			var n = Math.min(config.block_size, tokens.length - 1);
			var keys = [], values = [], losses = [];
			for (var l = 0; l < config.n_layer; l++) { keys.push([]); values.push([]); }

			// Forward 및 Loss(손실) 계산
			for (var p = 0; p < n; p++) {
				var logits = MicroGPT.Model.forward(tokens[p], p, keys, values, sd, config);
				var probs = MicroGPT.Model.softmax(logits);
				// 정답 확률의 로그값에 마이너스를 취함 (Cross Entropy Loss)
				losses.push(probs[tokens[p + 1]].log().neg());
			}

			var loss = losses[0];
			for (var i = 1; i < losses.length; i++) loss = loss.add(losses[i]);
			loss = loss.div(n);

			// Backward Pass (기울기 계산)
			for (var i = 0; i < params.length; i++) params[i].grad = 0;
			loss.backward();

			// Adam Optimizer Update: 기울기를 바탕으로 가중치를 최적의 방향으로 이동
			var lr = 0.01 * (1 - s / numSteps);
			for (var i = 0; i < params.length; i++) {
				m[i] = 0.9 * m[i] + 0.1 * params[i].grad;
				v[i] = 0.99 * v[i] + 0.01 * (params[i].grad * params[i].grad);
				var m_h = m[i] / (1 - Math.pow(0.9, s + 1));
				var v_h = v[i] / (1 - Math.pow(0.99, s + 1));
				params[i].data -= lr * m_h / (Math.sqrt(v_h) + 1e-8);
			}
			if (s % 10 === 0) process.stdout.write("Step " + s + " | Loss: " + loss.data.toFixed(4) + "\r");
		}
		console.log("\nTraining Completed.");
		if (callback) callback(sd, uchars, config, BOS);
	}
};

// ==========================================
// 5. Execution Controller (Usage Sample)
// 프로그램의 진입점입니다. 모델의 로드/학습/생성 흐름을 제어합니다.
// ==========================================

function main() {
	console.log("Checking for existing model...");
	var model = MicroGPT.Core.load();

	if (model) {
		// 모델이 있으면 로드하여 즉시 생성
		console.log("--- Generating Names from Loaded Model ---");
		var names = MicroGPT.Core.generate(10, model, 0.7);
		for (var i = 0; i < names.length; i++) console.log((i + 1) + ". " + names[i]);
	} else {
		// 모델이 없으면 학습 시퀀스 시작
		console.log("No model found. Starting training sequence...");
		MicroGPT.Core.train(300, function (sd, uchars, config, BOS) {
			MicroGPT.Core.save(sd, uchars, config);
			console.log("Model trained and saved. Run the script again to generate!");

			// 학습 직후 즉시 생성 테스트
			var modelData = { state_dict: sd, uchars: uchars, config: config, BOS: BOS };
			var names = MicroGPT.Core.generate(5, modelData, 0.7);
			console.log("Quick Sample:", names);
		});
	}
}

// 엔진 가동
main();