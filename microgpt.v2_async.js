/*
 * MicroGPT Node.js - Async Streaming & High Scalability
 * 기능: 비동기 데이터 스트리밍 학습, 메모리 효율 극대화, 대용량 파일 대응
 * -----------------------------------------------------------
 * [학습 가이드]
 * 1. Autograd: 신경망이 스스로 '어떻게 수정해야 할지' 계산하는 수학적 엔진
 * 2. Model: 데이터를 숫자 벡터로 변환하고 Transformer(GPT) 연산을 수행하는 몸체
 * 3. Core: 데이터를 읽어오고(Streaming), 실제로 지식을 저장(Save/Load)하는 제어 장치
 */

var fs = require('fs');
var https = require('https');
var path = require('path');
var readline = require('readline');

// 모델의 전역 설정 값들을 관리합니다.
global.MicroGPT = {
	Config: {
		seed: 42, // 재현성을 위한 난수 시드 (항상 같은 무작위 결과를 얻기 위함)
		inputPath: path.join(__dirname, 'input.txt'), // 학습용 데이터 파일 경로
		modelPath: path.join(__dirname, 'model_weights.json'), // 학습된 지식이 저장될 파일
		sourceUrl: 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt' // 기본 학습용 이름 데이터셋
	}
};

// ==========================================
// 1. Utils & Autograd (V2 최적화 유지)
// 인공지능의 '두뇌' 역할을 하는 수학 도구 모음입니다.
// ==========================================
MicroGPT.Utils = (function () {
	var seed = MicroGPT.Config.seed;
	return {
		// 결정론적 난수 생성기: 시드 기반으로 무작위 숫자를 만듭니다.
		random: function () {
			var x = Math.sin(seed++) * 10000;
			return x - Math.floor(x);
		},
		// 가우시안 정규 분포: 모델의 초기 가중치(지능의 초기 상태)를 자연스럽게 초기화합니다.
		gauss: function (mu, sigma) {
			var u1 = 0; while (u1 === 0) u1 = this.random();
			var u2 = 0; while (u2 === 0) u2 = this.random();
			var z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
			return z * sigma + mu;
		},
		// 확률 기반 선택: GPT가 다음 글자를 고를 때, 확률이 높은 글자를 더 자주 선택하게 합니다.
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

// 자동 미분(Autograd) 엔진: '오차 역전파'의 핵심입니다.
// 각 연산의 미분값(grad)을 계산하여 모델이 어느 방향으로 학습할지 결정합니다.
MicroGPT.Autograd = (function () {
	function Value(data, children, local_grads) {
		this.data = data; // 실제 수치값
		this.grad = 0;    // 이 값이 최종 결과(Loss)에 미치는 영향력 (기울기)
		this._children = children || null; // 이 값이 만들어지기 위해 참여한 부모 값들
		this._local_grads = local_grads || null; // 로컬 미분값
	}
	// 사칙연산 정의: 모든 연산 시 '미분 규칙'을 함께 저장합니다. (Chain Rule 준비)
	Value.prototype.add = function (other) {
		var o = (other instanceof Value) ? other : new Value(other);
		return new Value(this.data + o.data, [this, o], [1, 1]);
	};
	Value.prototype.mul = function (other) {
		var o = (other instanceof Value) ? other : new Value(other);
		return new Value(this.data * o.data, [this, o], [o.data, this.data]);
	};
	Value.prototype.pow = function (exp) {
		return new Value(Math.pow(this.data, exp), [this], [exp * Math.pow(this.data, exp - 1)]);
	};
	Value.prototype.log = function () {
		var val = this.data + 1e-10; // 0이 되는 것을 방지
		return new Value(Math.log(val), [this], [1 / val]);
	};
	Value.prototype.exp = function () {
		var res = Math.exp(this.data);
		return new Value(res, [this], [res]);
	};
	Value.prototype.relu = function () { // 활성화 함수: 음수는 버리고 양수만 전달하여 비선형성 부여
		return new Value(this.data > 0 ? this.data : 0, [this], [this.data > 0 ? 1 : 0]);
	};
	Value.prototype.div = function (other) {
		return this.mul((other instanceof Value ? other : new Value(other)).pow(-1));
	};
	Value.prototype.neg = function () { return this.mul(-1); };

	// 역전파 실행: 결과값에서 시작하여 입력값 방향으로 '누가 잘못했는지(grad)'를 전파합니다.
	Value.prototype.backward = function () {
		var topo = [], visited = new Set();
		function build(v) {
			if (visited.has(v)) return;
			visited.add(v);
			if (v._children) {
				for (var i = 0; i < v._children.length; i++) build(v._children[i]);
			}
			topo.push(v);
		}
		build(this);
		this.grad = 1; // 자기 자신에 대한 미분값은 1
		for (var i = topo.length - 1; i >= 0; i--) {
			var v = topo[i];
			if (!v._children) continue;
			for (var j = 0; j < v._children.length; j++) {
				v._children[j].grad += v._local_grads[j] * v.grad; // 연쇄 법칙(Chain Rule) 적용
			}
		}
	};
	return { Value: Value };
})();

// ==========================================
// 2. GPT Model Architecture (Logic)
// 실제 신경망의 구조(Layers)를 정의합니다.
// ==========================================
MicroGPT.Model = (function () {
	var Value = MicroGPT.Autograd.Value;
	var Utils = MicroGPT.Utils;
	return {
		// 가중치 행렬 생성: 지능이 담길 그릇(Matrix)을 만듭니다.
		createMatrix: function (rows, cols, std) {
			var mat = new Array(rows);
			for (var i = 0; i < rows; i++) {
				mat[i] = new Array(cols);
				for (var j = 0; j < cols; j++) mat[i][j] = new Value(Utils.gauss(0, std || 0.02));
			}
			return mat;
		},
		// 선형 변환 (Linear Layer): 행렬 곱셈을 통해 데이터를 고차원으로 해석합니다.
		linear: function (x, w) {
			var out = new Array(w.length);
			for (var i = 0; i < w.length; i++) {
				var row = w[i], sum = new Value(0);
				for (var j = 0; j < row.length; j++) sum = sum.add(row[j].mul(x[j]));
				out[i] = sum;
			}
			return out;
		},
		// 소프트맥스: 숫자들을 '확률(합계 100%)'로 변환합니다.
		softmax: function (logits) {
			var max_val = -Infinity;
			for (var i = 0; i < logits.length; i++) if (logits[i].data > max_val) max_val = logits[i].data;
			var exps = new Array(logits.length), sum_exps = new Value(0);
			for (var i = 0; i < logits.length; i++) {
				exps[i] = logits[i].add(-max_val).exp();
				sum_exps = sum_exps.add(exps[i]);
			}
			for (var i = 0; i < exps.length; i++) exps[i] = exps[i].div(sum_exps);
			return exps;
		},
		// RMSNorm: 데이터의 크기를 일정하게 유지하여 학습이 폭주하거나 사라지는 것을 방지합니다.
		rmsnorm: function (x) {
			var ss = new Value(0);
			for (var i = 0; i < x.length; i++) ss = ss.add(x[i].mul(x[i]));
			var inv_std = ss.div(x.length).add(1e-5).pow(-0.5);
			var out = new Array(x.length);
			for (var i = 0; i < x.length; i++) out[i] = x[i].mul(inv_std);
			return out;
		},
		// GPT Forward Pass: 토큰을 입력받아 다음 토큰이 무엇일지 예측하는 전체 과정
		forward: function (token_id, pos_id, keys, values, state_dict, config) {
			// 1. Embedding: 글자(token)와 위치(position)를 의미가 담긴 숫자 벡터로 변환
			var wte_row = state_dict.wte[token_id], wpe_row = state_dict.wpe[pos_id];
			var x = new Array(config.n_embd);
			for (var i = 0; i < config.n_embd; i++) x[i] = wte_row[i].add(wpe_row[i]);

			x = this.rmsnorm(x);

			// 2. Transformer Blocks: 데이터 간의 관계(Attention)와 특징 추출(MLP) 반복
			for (var l = 0; l < config.n_layer; l++) {
				var x_attn_res = x; // 잔차 연결을 위한 복사
				x = this.rmsnorm(x);

				// Attention: 현재 글자가 이전의 어떤 글자들과 연관이 있는지 계산
				var q = this.linear(x, state_dict['l' + l + '.wq']);
				var k = this.linear(x, state_dict['l' + l + '.wk']);
				var v = this.linear(x, state_dict['l' + l + '.wv']);
				keys[l].push(k); values[l].push(v);

				var heads_out = [];
				for (var h = 0; h < config.n_head; h++) {
					var start = h * config.head_dim, scores = new Array(keys[l].length);
					for (var t = 0; t < keys[l].length; t++) {
						var dot = new Value(0);
						for (var d = 0; d < config.head_dim; d++) dot = dot.add(q[start + d].mul(keys[l][t][start + d]));
						scores[t] = dot.div(Math.sqrt(config.head_dim));
					}
					var att = this.softmax(scores); // 중요도 점수 계산
					for (var d = 0; d < config.head_dim; d++) {
						var sum = new Value(0);
						for (var t = 0; t < att.length; t++) sum = sum.add(att[t].mul(values[l][t][start + d]));
						heads_out.push(sum);
					}
				}
				x = this.linear(heads_out, state_dict['l' + l + '.wo']);
				for (var i = 0; i < x.length; i++) x[i] = x[i].add(x_attn_res[i]); // 잔차 추가

				// MLP (Feed Forward): 추출된 관계를 바탕으로 더 복잡한 특징 학습
				var x_mlp_res = x;
				x = this.rmsnorm(x);
				x = this.linear(x, state_dict['l' + l + '.w1']);
				for (var i = 0; i < x.length; i++) x[i] = x[i].relu();
				x = this.linear(x, state_dict['l' + l + '.w2']);
				for (var i = 0; i < x.length; i++) x[i] = x[i].add(x_mlp_res[i]);
			}
			// 3. Head: 최종 결과를 다시 '글자 후보'들로 변환
			return this.linear(x, state_dict.lm_head);
		}
	};
})();

// ==========================================
// 3. Core Logic (Async Save/Load/Stream Train)
// 시스템의 실제 운영과 학습 프로세스를 담당합니다.
// ==========================================
MicroGPT.Core = {
	// 학습된 가중치를 JSON으로 저장
	save: function (state_dict, uchars, config) {
		var data = { config: config, uchars: uchars, weights: {} };
		for (var k in state_dict) {
			var mat = state_dict[k], rawMat = new Array(mat.length);
			for (var r = 0; r < mat.length; r++) {
				rawMat[r] = new Array(mat[r].length);
				for (var c = 0; c < mat[r].length; c++) rawMat[r][c] = mat[r][c].data;
			}
			data.weights[k] = rawMat;
		}
		fs.writeFileSync(MicroGPT.Config.modelPath, JSON.stringify(data));
		console.log("\n[Save] Model saved: " + MicroGPT.Config.modelPath);
	},

	// 저장된 지식을 다시 메모리로 로드
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

	// 비동기 스트리밍 학습 로직 (가장 중요한 부분)
	// 파일을 한 번에 메모리에 올리지 않고 '한 줄씩' 읽으며 학습하여 저사양 사양에서도 대용량 데이터를 처리합니다.
	trainAsync: async function (numSteps, callback) {
		var conf = MicroGPT.Config;

		// 1. 데이터셋 다운로드 체크: 데이터가 없으면 인터넷에서 가져옵니다.
		if (!fs.existsSync(conf.inputPath)) {
			console.log("Downloading dataset...");
			await new Promise((resolve) => {
				var file = fs.createWriteStream(conf.inputPath);
				https.get(conf.sourceUrl, (res) => {
					res.pipe(file).on('finish', () => { file.close(); resolve(); });
				});
			});
		}

		// 2. 어휘집(Vocabulary) 생성: 데이터에 어떤 글자들이 쓰였는지 확인합니다.
		console.log("Analyzing vocabulary...");
		var ucharsSet = new Set();
		const rl = readline.createInterface({ input: fs.createReadStream(conf.inputPath), crlfDelay: Infinity });
		for await (const line of rl) {
			for (const char of line.trim()) ucharsSet.add(char);
		}
		var uchars = Array.from(ucharsSet).sort();
		var BOS = uchars.length, vocab_size = BOS + 1; // BOS: 문장의 시작(Beginning Of Sentence) 토큰

		// 초소형 GPT 모델 설정
		var config = { n_layer: 1, n_embd: 16, block_size: 16, n_head: 4 };
		config.head_dim = config.n_embd / config.n_head;

		// 3. 모델 초기화: 텅 빈 뇌(임의의 난수 가중치)를 생성합니다.
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

		var params = [];
		for (var k in sd) {
			for (var r = 0; r < sd[k].length; r++) {
				for (var c = 0; c < sd[k][r].length; c++) params.push(sd[k][r][c]);
			}
		}

		// 최적화 도구 (Adam Optimizer): 효율적으로 공부하는 전략 (관성, 속도 조절)
		var m = new Float64Array(params.length), v = new Float64Array(params.length);
		var step = 0;

		// 4. 비동기 무한 스트리밍 루프: 파일을 '한 줄씩' 스트림으로 읽습니다.
		console.log("Starting Async Streaming Training...");
		while (step < numSteps) {
			const trainRl = readline.createInterface({ input: fs.createReadStream(conf.inputPath), crlfDelay: Infinity });

			for await (const line of trainRl) {
				if (step >= numSteps) break;

				var doc = line.trim();
				if (doc.length === 0) continue;

				// 텍스트를 숫자로 치환 (Tokenization)
				var tokens = [BOS];
				for (var i = 0; i < doc.length; i++) tokens.push(uchars.indexOf(doc[i]));
				tokens.push(BOS);

				var n = Math.min(config.block_size, tokens.length - 1);
				var keys = [], values = [], losses = [];
				for (var l = 0; l < config.n_layer; l++) { keys.push([]); values.push([]); }

				// Forward: 정답을 예측해보고 '손실(오차, Loss)'을 계산합니다.
				for (var p = 0; p < n; p++) {
					var logits = MicroGPT.Model.forward(tokens[p], p, keys, values, sd, config);
					var probs = MicroGPT.Model.softmax(logits);
					losses.push(probs[tokens[p + 1]].log().neg()); // 실제 정답 글자의 확률이 낮을수록 Loss가 커짐
				}

				var loss = losses[0];
				for (var i = 1; i < losses.length; i++) loss = loss.add(losses[i]);
				loss = loss.div(n);

				// Backward: 오차로부터 가중치를 어떻게 수정해야 할지 '기울기'를 계산합니다.
				for (var i = 0; i < params.length; i++) params[i].grad = 0;
				loss.backward();

				// Adam Update: 실제로 가중치(지능)를 업데이트합니다.
				var lr = 0.01 * (1 - step / numSteps); // 학습이 진행될수록 더 신중하게(작게) 업데이트
				for (var i = 0; i < params.length; i++) {
					m[i] = 0.9 * m[i] + 0.1 * params[i].grad;
					v[i] = 0.99 * v[i] + 0.01 * (params[i].grad * params[i].grad);
					var m_h = m[i] / (1 - Math.pow(0.9, step + 1));
					var v_h = v[i] / (1 - Math.pow(0.99, step + 1));
					params[i].data -= lr * m_h / (Math.sqrt(v_h) + 1e-8);
				}

				if (step % 5 === 0) {
					process.stdout.write(`Step ${step} | Loss: ${loss.data.toFixed(4)}\r`);
					// 중요: 이벤트 루프에 제어권을 넘겨 GC가 작동할 시간을 줌 (메모리 릭 방지)
					// 이 코드가 없으면 대용량 학습 시 메모리가 금방 꽉 찹니다.
					await new Promise(setImmediate);
				}
				step++;
			}
		}
		console.log("\nAsync Training Completed.");
		if (callback) callback(sd, uchars, config, BOS);
	},

	// 생성: 학습된 모델을 사용하여 새로운 텍스트(이름 등)를 만들어냅니다.
	generate: function (count, modelData, temperature) {
		var temp = temperature || 0.8, results = [];
		var sd = modelData.state_dict, cfg = modelData.config, uchars = modelData.uchars, BOS = modelData.BOS;

		for (var i = 0; i < count; i++) {
			var keys = [], values = [], sample = [], token_id = BOS;
			for (var l = 0; l < cfg.n_layer; l++) { keys.push([]); values.push([]); }
			// block_size만큼 글자를 하나씩 생성
			for (var pos = 0; pos < cfg.block_size; pos++) {
				var logits = MicroGPT.Model.forward(token_id, pos, keys, values, sd, cfg);
				var scaledLogits = new Array(logits.length);
				// Temperature: 창의성 조절 (높으면 의외의 결과, 낮으면 뻔한 결과)
				for (var j = 0; j < logits.length; j++) scaledLogits[j] = logits[j].div(temp);
				var probs = MicroGPT.Model.softmax(scaledLogits);
				var probData = new Array(probs.length);
				for (var j = 0; j < probs.length; j++) probData[j] = probs[j].data;

				// 확률적으로 다음 글자 선택
				var next = MicroGPT.Utils.choices([...Array(probs.length).keys()], probData);
				if (next === BOS) break; // 문장 종료 토큰이 나오면 멈춤
				sample.push(uchars[next]);
				token_id = next;
			}
			results.push(sample.join(''));
		}
		return results;
	}
};

// ==========================================
// 4. Execution Controller (Async Sample)
// 프로그램의 시작점입니다.
// ==========================================
async function main() {
	console.log("--- MicroGPT v2 (Async Streaming) ---");
	var model = MicroGPT.Core.load();

	if (model) {
		// 이미 공부한 지식이 있다면 불러와서 바로 테스트합니다.
		console.log("Loading existing model...");
		var names = MicroGPT.Core.generate(10, model, 0.7);
		names.forEach((n, i) => console.log(`${i + 1}. ${n}`));
	} else {
		// 지식이 없다면 대용량 파일도 읽을 수 있는 스트리밍 학습을 시작합니다.
		console.log("No model found. Starting Big-Data friendly training...");
		// 1TB 파일이어도 스트리밍으로 한 줄씩 읽어 학습합니다.
		await MicroGPT.Core.trainAsync(500, (sd, uchars, config, BOS) => {
			MicroGPT.Core.save(sd, uchars, config); // 학습 완료 후 저장
			console.log("Model saved. Ready to generate names!");

			var modelData = { state_dict: sd, uchars: uchars, config: config, BOS: BOS };
			console.log("Quick Test Result:", MicroGPT.Core.generate(5, modelData, 0.7));
		});
	}
}

main().catch(console.error);