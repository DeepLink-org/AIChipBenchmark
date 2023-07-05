import os
import time


def merge_dict(d1, d2):
    """
    Given d1={'a': {'a2': 1}}, d2={'a': {'a1': 1}}
    merge_dict(d1,d2) gives {'a': {'a2': 1, 'a1': 1}}
    """
    if not d1:
        return d2
    if not d2:
        return d1
    for k, v in d1.items():
        if d2.get(k, None) is not None:
            d2_val = d2[k]
            if isinstance(d1[k], dict) and isinstance(d2_val, dict):
                d1[k].update(d2_val)
            else:
                d1[k] = d2_val
        else:
            pass
    for k, v in d2.items():
        if d1.get(k, None) is None:
            d1[k] = d2[k]
        else:
            pass
    return d1


def test_merge_dict():
    assert merge_dict({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert merge_dict({"a": 1}, {"a": 2}) == {"a": 2}
    assert merge_dict({"a": {"a2": 1}}, d2={"a": {"a1": 1}}) == {"a": {"a2": 1, "a1": 1}}
    assert merge_dict({"a": {"a1": 1}}, d2={"a": {"a1": 2}}) == {"a": {"a1": 2}}
    assert merge_dict({"a": {"a1": 1}}, d2={"a": 2}) == {"a": 2}


def slurm_get_cur_node_count():
    user = os.environ.get("USER")
    process = os.popen('squeue -o "%.2D" --user ' + user)
    out_str = process.read()
    process.close()
    lines = out_str.split("\n")
    if len(lines) > 1:
        lines = lines[1:]
    cur_node_sum = 0
    for l in lines:
        if len(l) == 0:
            continue
        else:
            n_node = int(l.strip())
            cur_node_sum += n_node
    return cur_node_sum


def slurm_luanch_job(cmd, node, max_node=2):
    if os.environ.get("MAX_NODES", None) is not None:
        max_node = int(os.environ.get("MAX_NODES"))
    # get current using
    used = slurm_get_cur_node_count()
    printed = False
    while used + node > max_node:
        if not printed:
            print("waiting in queue ...")
            printed = True
        time.sleep(100)  # sleep 100 s
        used = slurm_get_cur_node_count()
    print("enqueue ...")
    os.system(cmd)


def test_slurm_launch():
    pass


print(slurm_get_cur_node_count())

test_merge_dict()
