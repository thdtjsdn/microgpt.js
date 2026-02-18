/*
 * MicroGPT Node.js - Async Streaming & High Scalability
 * 기능: 비동기 데이터 스트리밍 학습, 메모리 효율 극대화, 대용량 파일 대응
 */

var fs = require('fs');
var https = require('https');
var path = require('path');
var readline = require('readline');

var MicroGPT = {
	Config: {
		seed: 42,
		inputPath: path.join(__dirname, 'input.txt'),
		modelPath: path.join(__dirname, 'model_weights.json'),
		sourceUrl: 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
	}
};

// ==========================================
// 1. Utils & Autograd (V2 최적화 유지)
// ==========================================
MicroGPT.Utils = (function () {
	var seed = MicroGPT.Config.seed;
	return {
		random: function () {
			var x = Math.sin(seed++) * 10000;
			return x - Math.floor(x);
		},
		gauss: function (mu, sigma) {
			var u1 = 0; while (u1 === 0) u1 = this.random();
			var u2 = 0; while (u2 === 0) u2 = this.random();
			var z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
			return z * sigma + mu;
		},
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

MicroGPT.Autograd = (function () {
	function Value(data, children, local_grads) {
		this.data = data;
		this.grad = 0;
		this._children = children || null;
		this._local_grads = local_grads || null;
	}
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
		var val = this.data + 1e-10;
		return new Value(Math.log(val), [this], [1 / val]);
	};
	Value.prototype.exp = function () {
		var res = Math.exp(this.data);
		return new Value(res, [this], [res]);
	};
	Value.prototype.relu = function () {
		return new Value(this.data > 0 ? this.data : 0, [this], [this.data > 0 ? 1 : 0]);
	};
	Value.prototype.div = function (other) {
		return this.mul((other instanceof Value ? other : new Value(other)).pow(-1));
	};
	Value.prototype.neg = function () { return this.mul(-1); };
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
		this.grad = 1;
		for (var i = topo.length - 1; i >= 0; i--) {
			var v = topo[i];
			if (!v._children) continue;
			for (var j = 0; j < v._children.length; j++) {
				v._children[j].grad += v._local_grads[j] * v.grad;
			}
		}
	};
	return { Value: Value };
})();

// ==========================================
// 2. GPT Model Architecture (Logic)
// ==========================================
MicroGPT.Model = (function () {
	var Value = MicroGPT.Autograd.Value;
	var Utils = MicroGPT.Utils;
	return {
		createMatrix: function (rows, cols, std) {
			var mat = new Array(rows);
			for (var i = 0; i < rows; i++) {
				mat[i] = new Array(cols);
				for (var j = 0; j < cols; j++) mat[i][j] = new Value(Utils.gauss(0, std || 0.02));
			}
			return mat;
		},
		linear: function (x, w) {
			var out = new Array(w.length);
			for (var i = 0; i < w.length; i++) {
				var row = w[i], sum = new Value(0);
				for (var j = 0; j < row.length; j++) sum = sum.add(row[j].mul(x[j]));
				out[i] = sum;
			}
			return out;
		},
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
		rmsnorm: function (x) {
			var ss = new Value(0);
			for (var i = 0; i < x.length; i++) ss = ss.add(x[i].mul(x[i]));
			var inv_std = ss.div(x.length).add(1e-5).pow(-0.5);
			var out = new Array(x.length);
			for (var i = 0; i < x.length; i++) out[i] = x[i].mul(inv_std);
			return out;
		},
		forward: function (token_id, pos_id, keys, values, state_dict, config) {
			var wte_row = state_dict.wte[token_id], wpe_row = state_dict.wpe[pos_id];
			var x = new Array(config.n_embd);
			for (var i = 0; i < config.n_embd; i++) x[i] = wte_row[i].add(wpe_row[i]);
			x = this.rmsnorm(x);
			for (var l = 0; l < config.n_layer; l++) {
				var x_attn_res = x;
				x = this.rmsnorm(x);
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
					var att = this.softmax(scores);
					for (var d = 0; d < config.head_dim; d++) {
						var sum = new Value(0);
						for (var t = 0; t < att.length; t++) sum = sum.add(att[t].mul(values[l][t][start + d]));
						heads_out.push(sum);
					}
				}
				x = this.linear(heads_out, state_dict['l' + l + '.wo']);
				for (var i = 0; i < x.length; i++) x[i] = x[i].add(x_attn_res[i]);
				var x_mlp_res = x;
				x = this.rmsnorm(x);
				x = this.linear(x, state_dict['l' + l + '.w1']);
				for (var i = 0; i < x.length; i++) x[i] = x[i].relu();
				x = this.linear(x, state_dict['l' + l + '.w2']);
				for (var i = 0; i < x.length; i++) x[i] = x[i].add(x_mlp_res[i]);
			}
			return this.linear(x, state_dict.lm_head);
		}
	};
})();

// ==========================================
// 3. Core Logic (Async Save/Load/Stream Train)
// ==========================================
MicroGPT.Core = {
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
	trainAsync: async function (numSteps, callback) {
		var conf = MicroGPT.Config;

		// 1. 데이터셋 다운로드 체크
		if (!fs.existsSync(conf.inputPath)) {
			console.log("Downloading dataset...");
			await new Promise((resolve) => {
				var file = fs.createWriteStream(conf.inputPath);
				https.get(conf.sourceUrl, (res) => {
					res.pipe(file).on('finish', () => { file.close(); resolve(); });
				});
			});
		}

		// 2. 어휘집(Vocabulary) 생성 (대용량 대응을 위해 한 번 훑기)
		console.log("Analyzing vocabulary...");
		var ucharsSet = new Set();
		const rl = readline.createInterface({ input: fs.createReadStream(conf.inputPath), crlfDelay: Infinity });
		for await (const line of rl) {
			for (const char of line.trim()) ucharsSet.add(char);
		}
		var uchars = Array.from(ucharsSet).sort();
		var BOS = uchars.length, vocab_size = BOS + 1;

		var config = { n_layer: 1, n_embd: 16, block_size: 16, n_head: 4 };
		config.head_dim = config.n_embd / config.n_head;

		// 3. 모델 초기화
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

		var m = new Float64Array(params.length), v = new Float64Array(params.length);
		var step = 0;

		// 4. 비동기 무한 스트리밍 루프
		console.log("Starting Async Streaming Training...");
		while (step < numSteps) {
			const trainRl = readline.createInterface({ input: fs.createReadStream(conf.inputPath), crlfDelay: Infinity });

			for await (const line of trainRl) {
				if (step >= numSteps) break;

				var doc = line.trim();
				if (doc.length === 0) continue;

				var tokens = [BOS];
				for (var i = 0; i < doc.length; i++) tokens.push(uchars.indexOf(doc[i]));
				tokens.push(BOS);

				var n = Math.min(config.block_size, tokens.length - 1);
				var keys = [], values = [], losses = [];
				for (var l = 0; l < config.n_layer; l++) { keys.push([]); values.push([]); }

				// Forward
				for (var p = 0; p < n; p++) {
					var logits = MicroGPT.Model.forward(tokens[p], p, keys, values, sd, config);
					var probs = MicroGPT.Model.softmax(logits);
					losses.push(probs[tokens[p + 1]].log().neg());
				}

				var loss = losses[0];
				for (var i = 1; i < losses.length; i++) loss = loss.add(losses[i]);
				loss = loss.div(n);

				// Backward
				for (var i = 0; i < params.length; i++) params[i].grad = 0;
				loss.backward();

				// Adam Update
				var lr = 0.01 * (1 - step / numSteps);
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
					await new Promise(setImmediate);
				}
				step++;
			}
		}
		console.log("\nAsync Training Completed.");
		if (callback) callback(sd, uchars, config, BOS);
	},

	generate: function (count, modelData, temperature) {
		var temp = temperature || 0.8, results = [];
		var sd = modelData.state_dict, cfg = modelData.config, uchars = modelData.uchars, BOS = modelData.BOS;

		for (var i = 0; i < count; i++) {
			var keys = [], values = [], sample = [], token_id = BOS;
			for (var l = 0; l < cfg.n_layer; l++) { keys.push([]); values.push([]); }
			for (var pos = 0; pos < cfg.block_size; pos++) {
				var logits = MicroGPT.Model.forward(token_id, pos, keys, values, sd, cfg);
				var scaledLogits = new Array(logits.length);
				for (var j = 0; j < logits.length; j++) scaledLogits[j] = logits[j].div(temp);
				var probs = MicroGPT.Model.softmax(scaledLogits);
				var probData = new Array(probs.length);
				for (var j = 0; j < probs.length; j++) probData[j] = probs[j].data;
				var next = MicroGPT.Utils.choices([...Array(probs.length).keys()], probData);
				if (next === BOS) break;
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
// ==========================================
async function main() {
	console.log("--- MicroGPT v3 (Async Streaming) ---");
	var model = MicroGPT.Core.load();

	if (model) {
		console.log("Loading existing model...");
		var names = MicroGPT.Core.generate(10, model, 0.7);
		names.forEach((n, i) => console.log(`${i + 1}. ${n}`));
	} else {
		console.log("No model found. Starting Big-Data friendly training...");
		// 1TB 파일이어도 스트리밍으로 한 줄씩 읽어 학습합니다.
		await MicroGPT.Core.trainAsync(500, (sd, uchars, config, BOS) => {
			MicroGPT.Core.save(sd, uchars, config);
			console.log("Model saved. Ready to generate names!");

			var modelData = { state_dict: sd, uchars: uchars, config: config, BOS: BOS };
			console.log("Quick Test Result:", MicroGPT.Core.generate(5, modelData, 0.7));
		});
	}
}

main().catch(console.error);