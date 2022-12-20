def ordinary_points(n):
    """ordinary points are just pairs (x, y) where x and y
    are both between 0 and n - 1"""
    return [(x, y) for x in range(n) for y in range(n)]
    
def points_at_infinity(n):
    """infinite points are just the numbers 0 to n - 1
    (corresponding to the infinity where lines with that slope meet)
    and infinity infinity (where vertical lines meet)"""
    return list(range(n)) + [u"∞"]

def all_points(n):
    return ordinary_points(n) + points_at_infinity(n)

def ordinary_line(m, b, n):
    """returns the ordinary line through (0, b) with slope m
    in the finite projective plan of degree n
    includes 'infinity m'"""
    return [(x, (m * x + b) % n) for x in range(n)] + [m]

def vertical_line(x, n):
    """returns the vertical line with the specified x-coordinate
    in the finite projective plane of degree n
    includes 'infinity infinity'"""
    return [(x, y) for y in range(n)] + [u"∞"]
    
def line_at_infinity(n):
    """the line at infinity just contains the points at infinity"""
    return points_at_infinity(n)

def all_lines(n):
    return ([ordinary_line(m, b, n) for m in range(n) for b in range(n)] +
            [vertical_line(x, n) for x in range(n)] +
            [line_at_infinity(n)])
def make_deck(n, pics):
    points = all_points(n)

    # create a mapping from point to pic
    mapping = { point : pic 
                for point, pic in zip(points, pics) }

    # and return the remapped cards
    return [map(mapping.get, line) for line in all_lines(n)]

def test_deck(deck):
    for c in deck:
        for f in c:
            if f is None:
                return False
    return True

def deck_options(faces):
    face_per_deck_options = []
    for i in range(3,# Min Face Per Card
                   9+1,# Max Face Per Card
                ):
        if test_deck(make_deck(i, faces)):
            face_per_deck_options.append(i)
    return face_per_deck_options