/*
 * MicroGPT Node.js - High Performance & Full Integration
 * 기능: Autograd, GPT Architecture, Adam Optimizer, Save/Load, Generator
 */

var fs = require('fs');
var https = require('https');
var path = require('path');

// ==========================================
// 0. Configuration & Path Management
// ==========================================
var MicroGPT = {
	Config: {
		seed: 42,
		inputPath: path.join(__dirname, 'input.txt'),
		modelPath: path.join(__dirname, 'model_weights.json'),
		sourceUrl: 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
	}
};

// ==========================================
// 1. Utils (Performance Optimized)
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

// ==========================================
// 2. Autograd Engine (Value Object)
// ==========================================
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
// 3. GPT Model Architecture (Optimized For-loops)
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
				var row = w[i];
				var sum = new Value(0);
				for (var j = 0; j < row.length; j++) sum = sum.add(row[j].mul(x[j]));
				out[i] = sum;
			}
			return out;
		},
		softmax: function (logits) {
			var max_val = -Infinity;
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
		rmsnorm: function (x) {
			var ss = new Value(0);
			for (var i = 0; i < x.length; i++) ss = ss.add(x[i].mul(x[i]));
			var inv_std = ss.div(x.length).add(1e-5).pow(-0.5);
			var out = new Array(x.length);
			for (var i = 0; i < x.length; i++) out[i] = x[i].mul(inv_std);
			return out;
		},
		forward: function (token_id, pos_id, keys, values, state_dict, config) {
			var wte_row = state_dict.wte[token_id];
			var wpe_row = state_dict.wpe[pos_id];
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
					var start = h * config.head_dim;
					var scores = new Array(keys[l].length);
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
// 4. Core Logic (Save, Load, Train, Generate)
// ==========================================
MicroGPT.Core = {
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
	},

	train: function (numSteps, callback) {
		var conf = MicroGPT.Config;
		if (!fs.existsSync(conf.inputPath)) {
			console.log("Downloading dataset...");
			var file = fs.createWriteStream(conf.inputPath);
			https.get(conf.sourceUrl, function (res) {
				res.pipe(file).on('finish', function () { file.close(); MicroGPT.Core.train(numSteps, callback); });
			});
			return;
		}

		var text = fs.readFileSync(conf.inputPath, 'utf8');
		var docs = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
		var uchars = Array.from(new Set(docs.join(''))).sort();
		var BOS = uchars.length, vocab_size = BOS + 1;

		var config = { n_layer: 1, n_embd: 16, block_size: 16, n_head: 4 };
		config.head_dim = config.n_embd / config.n_head;

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
		console.log("Training Start...");

		for (var s = 0; s < numSteps; s++) {
			var doc = docs[Math.floor(MicroGPT.Utils.random() * docs.length)];
			var tokens = [BOS];
			for (var i = 0; i < doc.length; i++) tokens.push(uchars.indexOf(doc[i]));
			tokens.push(BOS);

			var n = Math.min(config.block_size, tokens.length - 1);
			var keys = [], values = [], losses = [];
			for (var l = 0; l < config.n_layer; l++) { keys.push([]); values.push([]); }

			for (var p = 0; p < n; p++) {
				var logits = MicroGPT.Model.forward(tokens[p], p, keys, values, sd, config);
				var probs = MicroGPT.Model.softmax(logits);
				losses.push(probs[tokens[p + 1]].log().neg());
			}

			var loss = losses[0];
			for (var i = 1; i < losses.length; i++) loss = loss.add(losses[i]);
			loss = loss.div(n);

			for (var i = 0; i < params.length; i++) params[i].grad = 0;
			loss.backward();

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
// ==========================================

function main() {
	console.log("Checking for existing model...");
	var model = MicroGPT.Core.load();

	if (model) {
		console.log("--- Generating Names from Loaded Model ---");
		var names = MicroGPT.Core.generate(10, model, 0.7);
		for (var i = 0; i < names.length; i++) console.log((i + 1) + ". " + names[i]);
	} else {
		console.log("No model found. Starting training sequence...");
		// 학습 단계를 500단계 정도로 늘리면 성능이 더 좋아집니다.
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

// Start the engine
main();