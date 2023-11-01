def naive_iterator(n):
    context_id = 0
    for i in range(n):
        for j in range(n):
            k = {"context_id": context_id}
            yield i, j, k
            context_id += 1


def zig_zag_horizontal(n):
    context_id = 0
    for i in range(n):
        inner = list(range(n))
        if i % 2 == 1:
            inner = reversed(inner)
        for j in inner:
            k = {"context_id": context_id}
            yield i, j, k
            context_id += 1


def naive_iterator_2s(n):
    context_id = 0
    for i in range(n):
        for j in range(n):
            k = {"context_id": context_id, "mem_only": True}
            yield i, j, k
            context_id += 1
    for i in range(n):
        for j in range(n):
            k = {"context_id": context_id}
            yield i, j, k
            context_id += 1


REGISTRY = dict(
    naive=naive_iterator,
    zig_zag_horizontal=zig_zag_horizontal,
    naive_two_stream=naive_iterator_2s,
)


def build_tiling(n, method):
    if method in REGISTRY:
        return REGISTRY[method](n)
    else:
        raise NotImplementedError
