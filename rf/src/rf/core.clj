(ns rf.core
  (:require [clojure.zip :as zip]))

(def sample [{:a 0 :b 0 :c 0 :d 0 :class :0} {:a 1 :b 0 :c 0 :d 0 :class :8}
             {:a 0 :b 0 :c 0 :d 1 :class :1} {:a 1 :b 0 :c 0 :d 1 :class :9}
             {:a 0 :b 0 :c 1 :d 0 :class :2} {:a 1 :b 0 :c 1 :d 0 :class :a}
             {:a 0 :b 0 :c 1 :d 1 :class :3} {:a 1 :b 0 :c 1 :d 1 :class :b}
             {:a 0 :b 1 :c 0 :d 0 :class :4} {:a 1 :b 1 :c 0 :d 0 :class :c}
             {:a 0 :b 1 :c 0 :d 1 :class :5} {:a 1 :b 1 :c 0 :d 1 :class :d}
             {:a 0 :b 1 :c 1 :d 0 :class :6} {:a 1 :b 1 :c 1 :d 0 :class :e}
             {:a 0 :b 1 :c 1 :d 1 :class :7} {:a 1 :b 1 :c 1 :d 1 :class :f}])

(defn bootstrap [dataset]
  (let [n (count dataset)
        s (repeatedly n #(rand-int n))
        d (set (map #(nth dataset %) s))
        o (set (filter (comp not d) dataset))]
    {:dataset d :oob o}))

(defn select [[v s] dataset]
  (let [l (reduce (fn [acc t] (if (<= (get t v) s) (conj acc t) acc))
            #{} dataset)
        r (set (filter (comp not l) dataset))]
    [l r]))

(defn entropy [dataset]
  (let [n (count dataset)
        p (map #(/ % n) (vals (frequencies (map :class dataset))))
        l #(/ (Math/log %) (Math/log 2))]
    (* -1 (reduce #(+ %1 (* %2 (l %2))) 0.0 p))))

(defn splits [dataset mtry]
  (let [vs (filter (partial not= :class) (keys (first dataset)))
        n (count vs)
        vs (if (<= mtry n) (take mtry (shuffle vs)) vs)]
    (reduce (fn [acc v] (reduce (fn [acc t] (conj acc [v (get t v)])) acc dataset)) #{} vs)))

(defn best-split [dataset oob mtry]
  (let [[_ s l r] (first (sort (map (fn [s] (let [[l r] (select s dataset)]
                                              [(entropy l) s l r])) (splits dataset mtry))))]
    (let [[oobl oobr] (select s oob)] [s (set l) (set r) (set oobl) (set oobr)])))

(defn terminal? [dataset] (= (entropy dataset) 0.0))

(defn extend-node [{:keys [dataset oob] :as node} mtry]
  (if (terminal? dataset)
    (merge node {:class (or (first (first (sort-by val > (frequencies (map :class dataset))))) :unknown)})
    (let [[s l r oobl oobr] (best-split dataset oob mtry)]
      (merge node {:criterion s :left  {:dataset l :oob oobl}
                                :right {:dataset r :oob oobr}}))))

(defn rf-train [dataset ntree mtry]
  (set (pmap (fn [sample]
            (loop [loc (zip/zipper (fn [node] true)
                                   #(if (:left %) (seq [(:left %) (:right %)]))
                                   (fn [node children]
                                     (with-meta (merge node {:left  (first children)
                                                             :right (second children)}) (meta node)))
                                   sample)]
              (if (zip/end? loc) (zip/root loc)
                (recur (zip/next (zip/edit loc extend-node mtry))))))
          (repeatedly ntree #(bootstrap dataset)))))

(defn rf-predict [forest features]
  (let [eval-tree (fn eval-tree [tree]
                    (if (:class tree) (:class tree)
                      (if (let [[v s] (:criterion tree)] (<= (get features v) s))
                        (eval-tree (:left tree)) (eval-tree (:right tree)))))]
    (first (first (sort-by val > (frequencies (filter (partial not= :unknown) (pmap eval-tree forest))))))))

(defn tree-error [tree]
  (let [cls (:class tree)]
    (cond (nil? cls) (+ (tree-error (:left tree)) (tree-error (:right tree)))
          (= cls :unknown) 0 :else (count (filter (partial = cls) (:oob tree))))))

(defn rf-error [forest] (/ (reduce + 0 (pmap #(/ (tree-error %) (count (:oob %))) forest)) (count forest)))

(def rf (rf-train sample 1 3))

(every? #(= (rf-predict rf (dissoc % :class)) (:class %)) sample)

(rf-error rf)