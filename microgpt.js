/*
 * MicroGPT Node.js Port (Final Optimized Version)
 * 포트: Prototype & Namespace architecture
 * 기능: 자동 미분(Autograd), GPT 아키텍처, Adam 최적화, 모델 저장/로드 및 생성
 */

var fs = require('fs');
var https = require('https');
var path = require('path');

// ==========================================
// 0. Namespace & Global Config
// ==========================================
var MicroGPT = {
	Config: {
		seed: 42,
		inputPath: path.join(__dirname, 'input.txt'),
		modelPath: path.join(__dirname, 'model_weights.json'), // 공통 경로 지정
		sourceUrl: 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
	},
	Utils: {},
	Autograd: {},
	Model: {},
	Core: {}
};

// ==========================================
// 1. Utils (Math & Helpers)
// ==========================================
MicroGPT.Utils = (function () {
	var seed = MicroGPT.Config.seed;
	function random() {
		var x = Math.sin(seed++) * 10000;
		return x - Math.floor(x);
	}
	function gauss(mu, sigma) {
		var u1 = 0, u2 = 0;
		while (u1 === 0) u1 = random();
		while (u2 === 0) u2 = random();
		var z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
		return z * sigma + mu;
	}
	function shuffle(array) {
		for (var i = array.length - 1; i > 0; i--) {
			var j = Math.floor(random() * (i + 1));
			var temp = array[i];
			array[i] = array[j];
			array[j] = temp;
		}
		return array;
	}
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
// 2. Autograd Engine
// ==========================================
MicroGPT.Autograd.Value = (function () {
	function Value(data, children, local_grads) {
		this.data = data;
		this.grad = 0;
		this._children = children || [];
		this._local_grads = local_grads || [];
	}
	Value.prototype.add = function (other) {
		other = (other instanceof Value) ? other : new Value(other);
		return new Value(this.data + other.data, [this, other], [1, 1]);
	};
	Value.prototype.mul = function (other) {
		other = (other instanceof Value) ? other : new Value(other);
		return new Value(this.data * other.data, [this, other], [other.data, this.data]);
	};
	Value.prototype.pow = function (exp) {
		return new Value(Math.pow(this.data, exp), [this], [exp * Math.pow(this.data, exp - 1)]);
	};
	Value.prototype.log = function () {
		return new Value(Math.log(this.data + 1e-10), [this], [1 / (this.data + 1e-10)]);
	};
	Value.prototype.exp = function () {
		var res = Math.exp(this.data);
		return new Value(res, [this], [res]);
	};
	Value.prototype.relu = function () {
		return new Value(Math.max(0, this.data), [this], [(this.data > 0 ? 1 : 0)]);
	};
	Value.prototype.neg = function () { return this.mul(-1); };
	Value.prototype.sub = function (other) { return this.add(other instanceof Value ? other.neg() : new Value(-other)); };
	Value.prototype.div = function (other) { return this.mul(other instanceof Value ? other.pow(-1) : new Value(other).pow(-1)); };

	Value.prototype.backward = function () {
		var topo = [], visited = new Set();
		function build_topo(v) {
			if (!visited.has(v)) {
				visited.add(v);
				v._children.forEach(build_topo);
				topo.push(v);
			}
		}
		build_topo(this);
		this.grad = 1;
		for (var i = topo.length - 1; i >= 0; i--) {
			var v = topo[i];
			for (var j = 0; j < v._children.length; j++) {
				v._children[j].grad += v._local_grads[j] * v.grad;
			}
		}
	};
	return Value;
})();

// ==========================================
// 3. GPT Model Architecture
// ==========================================
MicroGPT.Model = (function () {
	var Value = MicroGPT.Autograd.Value;
	var Utils = MicroGPT.Utils;

	function createMatrix(rows, cols, std) {
		var mat = [];
		for (var i = 0; i < rows; i++) {
			var row = [];
			for (var j = 0; j < cols; j++) row.push(new Value(Utils.gauss(0, std || 0.02)));
			mat.push(row);
		}
		return mat;
	}

	function linear(x, w) {
		return w.map(function (row) {
			var sum = new Value(0);
			for (var i = 0; i < row.length; i++) sum = sum.add(row[i].mul(x[i]));
			return sum;
		});
	}

	function softmax(logits) {
		var max_val = logits.reduce(function (m, v) { return Math.max(m, v.data); }, -Infinity);
		var exps = logits.map(function (v) { return v.sub(max_val).exp(); });
		var sum_exps = exps.reduce(function (a, b) { return a.add(b); }, new Value(0));
		return exps.map(function (e) { return e.div(sum_exps); });
	}

	function rmsnorm(x) {
		var ss = x.reduce(function (a, b) { return a.add(b.mul(b)); }, new Value(0)).div(x.length);
		var inv_std = ss.add(1e-5).pow(-0.5);
		return x.map(function (v) { return v.mul(inv_std); });
	}

	function forward(token_id, pos_id, keys, values, state_dict, config) {
		var x = state_dict.wte[token_id].map(function (v, i) { return v.add(state_dict.wpe[pos_id][i]); });
		x = rmsnorm(x);

		for (var l = 0; l < config.n_layer; l++) {
			var x_res = x;
			x = rmsnorm(x);
			// Attention
			var q = linear(x, state_dict['l'+l+'.wq']);
			var k = linear(x, state_dict['l'+l+'.wk']);
			var v = linear(x, state_dict['l'+l+'.wv']);
			keys[l].push(k); values[l].push(v);

			var heads = [];
			for (var h = 0; h < config.n_head; h++) {
				var start = h * config.head_dim, end = start + config.head_dim;
				var qh = q.slice(start, end);
				var kh_hist = keys[l].map(function(ki) { return ki.slice(start, end); });
				var vh_hist = values[l].map(function(vi) { return vi.slice(start, end); });

				var scores = kh_hist.map(function(kh) {
					var dot = new Value(0);
					for(var i=0; i<qh.length; i++) dot = dot.add(qh[i].mul(kh[i]));
					return dot.div(Math.sqrt(config.head_dim));
				});
				var att = softmax(scores);
				var hout = Array(config.head_dim).fill(0).map(function(_, i) {
					var sum = new Value(0);
					att.forEach(function(a, t) { sum = sum.add(a.mul(vh_hist[t][i])); });
					return sum;
				});
				heads = heads.concat(hout);
			}
			x = linear(heads, state_dict['l'+l+'.wo']).map(function(v, i) { return v.add(x_res[i]); });

			// MLP
			x_res = x;
			x = rmsnorm(x);
			x = linear(x, state_dict['l'+l+'.w1']).map(function(v) { return v.relu(); });
			x = linear(x, state_dict['l'+l+'.w2']).map(function(v, i) { return v.add(x_res[i]); });
		}
		return linear(x, state_dict.lm_head);
	}

	return { createMatrix: createMatrix, forward: forward, softmax: softmax };
})();

// ==========================================
// 4. Core Logic (Train, Save, Load, Generate)
// ==========================================
MicroGPT.Core = {
	saveModel: function(state_dict, uchars, config, filePath) {
		var data = { config: config, uchars: uchars, weights: {} };
		for (var k in state_dict) {
			data.weights[k] = state_dict[k].map(function(row) {
				return row.map(function(v) { return v.data; });
			});
		}
		fs.writeFileSync(filePath || MicroGPT.Config.modelPath, JSON.stringify(data));
		console.log("\n[Success] Model saved to: " + (filePath || MicroGPT.Config.modelPath));
	},

	loadModel: function(filePath) {
		var p = filePath || MicroGPT.Config.modelPath;
		if (!fs.existsSync(p)) return null;
		var data = JSON.parse(fs.readFileSync(p, 'utf8'));
		var state_dict = {};
		for (var k in data.weights) {
			state_dict[k] = data.weights[k].map(function(row) {
				return row.map(function(d) { return new MicroGPT.Autograd.Value(d); });
			});
		}
		console.log("[Success] Model loaded.");
		return { state_dict: state_dict, uchars: data.uchars, config: data.config, BOS: data.uchars.length };
	},

	generate: function(count, modelData, temperature) {
		var temp = temperature || 0.8;
		var results = [];
		var sd = modelData.state_dict, cfg = modelData.config, uchars = modelData.uchars, BOS = modelData.BOS;

		for (var i = 0; i < count; i++) {
			var keys = [], values = [], sample = [], token_id = BOS;
			for(var l=0; l<cfg.n_layer; l++) { keys.push([]); values.push([]); }

			for (var pos = 0; pos < cfg.block_size; pos++) {
				var logits = MicroGPT.Model.forward(token_id, pos, keys, values, sd, cfg);
				var probs = MicroGPT.Model.softmax(logits.map(function(l) { return l.div(temp); }));
				var next = MicroGPT.Utils.choices([...Array(probs.length).keys()], probs.map(p => p.data));
				if (next === BOS) break;
				sample.push(uchars[next]);
				token_id = next;
			}
			results.push(sample.join(''));
		}
		return results;
	},

	train: function(numSteps, callback) {
		var conf = MicroGPT.Config;
		if (!fs.existsSync(conf.inputPath)) {
			console.log("Downloading dataset...");
			var file = fs.createWriteStream(conf.inputPath);
			https.get(conf.sourceUrl, function(res) {
				res.pipe(file).on('finish', function() { file.close(); MicroGPT.Core.train(numSteps, callback); });
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
		for(var i=0; i<config.n_layer; i++) {
			sd['l'+i+'.wq'] = MicroGPT.Model.createMatrix(config.n_embd, config.n_embd);
			sd['l'+i+'.wk'] = MicroGPT.Model.createMatrix(config.n_embd, config.n_embd);
			sd['l'+i+'.wv'] = MicroGPT.Model.createMatrix(config.n_embd, config.n_embd);
			sd['l'+i+'.wo'] = MicroGPT.Model.createMatrix(config.n_embd, config.n_embd);
			sd['l'+i+'.w1'] = MicroGPT.Model.createMatrix(config.n_embd * 4, config.n_embd);
			sd['l'+i+'.w2'] = MicroGPT.Model.createMatrix(config.n_embd, config.n_embd * 4);
		}

		var params = [];
		for(var k in sd) sd[k].forEach(row => row.forEach(p => params.push(p)));

		var m = Array(params.length).fill(0), v = Array(params.length).fill(0);
		console.log("Training (" + numSteps + " steps)...");

		for (var s = 0; s < numSteps; s++) {
			var doc = docs[s % docs.length];
			var tokens = [BOS].concat(doc.split('').map(c => uchars.indexOf(c))).concat([BOS]);
			var n = Math.min(config.block_size, tokens.length - 1);
			var keys = [], values = [], losses = [];
			for(var l=0; l<config.n_layer; l++) { keys.push([]); values.push([]); }

			for (var p = 0; p < n; p++) {
				var logits = MicroGPT.Model.forward(tokens[p], p, keys, values, sd, config);
				var probs = MicroGPT.Model.softmax(logits);
				losses.push(probs[tokens[p+1]].log().neg());
			}
			var loss = losses.reduce((a, b) => a.add(b), new MicroGPT.Autograd.Value(0)).div(n);
			params.forEach(p => p.grad = 0);
			loss.backward();

			// Adam Update
			var lr = 0.01 * (1 - s/numSteps);
			for(var i=0; i<params.length; i++) {
				m[i] = 0.9 * m[i] + 0.1 * params[i].grad;
				v[i] = 0.99 * v[i] + 0.01 * Math.pow(params[i].grad, 2);
				var m_h = m[i] / (1 - Math.pow(0.9, s + 1));
				var v_h = v[i] / (1 - Math.pow(0.99, s + 1));
				params[i].data -= lr * m_h / (Math.sqrt(v_h) + 1e-8);
			}
			if (s % 10 === 0) process.stdout.write("Step " + s + " | Loss: " + loss.data.toFixed(4) + "\r");
		}
		console.log("\nDone.");
		if(callback) callback(sd, uchars, config, BOS);
	}
};

// ==========================================
// 5. Usage Example (사용법)
// ==========================================

// 1) 학습 후 저장하기 예시
/*
MicroGPT.Core.train(100, function(sd, uchars, config, BOS) {
	MicroGPT.Core.saveModel(sd, uchars, config);

	// 생성 테스트
	var names = MicroGPT.Core.generate(5, {state_dict: sd, uchars: uchars, config: config, BOS: BOS});
	console.log("Generated:", names);
});
*/

// 2) 저장된 모델 불러와서 생성하기 예시
var savedData = MicroGPT.Core.loadModel();
if (savedData) {
	console.log("학습된 모델로 이름을 생성합니다:");
	var names = MicroGPT.Core.generate(10, savedData, 0.7);
	names.forEach((n, i) => console.log((i+1) + ". " + n));
} else {
	console.log("저장된 모델이 없습니다. 먼저 학습(train)을 진행해주세요.");
	// 모델이 없을 경우 자동으로 학습 시작 (샘플)
	MicroGPT.Core.train(200, function(sd, uchars, config, BOS) {
		MicroGPT.Core.saveModel(sd, uchars, config);
		console.log("학습 완료 및 모델 저장됨. 다시 실행하면 로드하여 생성합니다.");
	});
}