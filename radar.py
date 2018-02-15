"""Constructs a radar plot using matplotlib"""

import os
import re
from textwrap import wrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from matplotlib.patches import Polygon, RegularPolygon
from matplotlib.collections import PatchCollection, LineCollection, PolyCollection




# FIXME: Change so this is not a constant?
IGNORE = None

class RadarPlot(object):
    """Tools to construct a radar plot using matplotlib

    Attributes:
        vertices (list): names of the vertices/slices of the plot
        num_vertices (int): number of vertices/sides
        radius (float): radius of the radar plot
        maxval (int): the highest value a spine can have
        internal (int): the interval to draw ticks on each spine
        fill (bool): specifies whether values fill the slice or mark the spine
            (default is to fill the slice)
        colors (list): list of colors for plotting multiple fields
    """
    colors = [
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
    ]


    def __init__(self, vertices, maxval=5, interval=1, fill=True):
        self.vertices = vertices
        self.num_vertices = len(vertices)
        self.maxval = float(maxval)
        self.interval = interval
        self.fill = fill
        self.radius = 0.5  # the radius corresponds to the center of the figure
        self._slices = []
        self._fields = []


    def grid(self):
        """Constructs and returns the grid for this plot

        Returns:
            PatchCollection containg the grid
        """
        grid = [RegularPolygon((self.radius, self.radius),
                               self.num_vertices, self.radius)]
        num_ticks = int(self.maxval) / self.interval
        for i in xrange(num_ticks):
            val = float(i) / num_ticks * self.radius
            grid.append(RegularPolygon((self.radius, self.radius),
                                       self.num_vertices, val))
        facecolors = [(1, 1, 1, 0)] * len(grid)
        edgecolors = ['k'] + [(0, 0, 0, 0.4)] * len(grid[1:])
        linewidths = [0.5] + [0.2] * len(grid[1:])
        return PatchCollection(grid,
                               facecolors=facecolors,
                               edgecolors=edgecolors,
                               linewidths=linewidths,
                               zorder=1)


    def spines(self):
        """Constructs and returns the collection of spines for this plot

        Returns:
            LineCollection containing the radial spines
        """
        spines = []
        for i, _ in enumerate(self.vertices):
            x, y = self.get_xy(self.radius, i)
            spines.append([(self.radius, self.radius), (x, y)])
        colors = [(0, 0, 0, 0.4)] * len(spines)
        linewidths = [0.2 for s in spines]
        return LineCollection(spines,
                              colors=colors,
                              linewidths=linewidths,
                              zorder=4)


    def labels(self):
        """Constructs and returns the labels for this plot

        Returns:
            List of tuples containing the labels and positioning information
        """
        labels = []
        for i, label in enumerate(self.vertices):
            label = preplabel(label)
            # Slide the label to the middle of the slice if fill
            if self.fill:
                i += 0.5
            x, y = self.get_xy(self.radius * 1.02, i)
            theta = self.get_theta(i)
            rotation = theta * 57.3 + (90 if y <= self.radius else -90)
            va = 'top' if y < self.radius else 'bottom'
            labels.append(((x, y, '\n'.join(wrap(label, 16)), rotation, va)))
        return labels


    def slices(self):
        """Returns the fields that have been added to this plot

        Returns:
            PolyCollection containing the slices
        """
        facecolors = [(0, 0, 0, 0.05)] * len(self._slices)
        return PolyCollection(self._slices, facecolors=facecolors, zorder=2)


    def fields(self, labels=None, linewidth=0):
        """Returns the fields that have been added to this plot

        Args:
            labels (list): labels for the fields in the plot

        Returns:
            PolyCollection containing the fields
        """
        if labels is not None:
            assert len(labels) == len(self._fields)
        colors = self.colors[:len(self._fields)]
        facecolors = [(r, g, b, 0.25) for r, g, b in colors]
        edgecolors = [(r, g, b, 1) for r, g, b in colors]
        linewidths = [linewidth] * len(self._fields)
        fields = []
        for i, field in enumerate(self._fields):
            field = Polygon(field,
                            fc=facecolors[i],
                            ec=edgecolors[i],
                            lw=linewidths[i],
                            zorder=3)
            if labels is not None:
                field.set_label(labels[i])
            fields.append(field)
        # Removed PolyCollection so each field can have its own label ~AM
        return fields


    def get_theta(self, spine=0):
        """Returns the angle of a spine

        Args:
            spine (int): the index of the spine

        Returns:
            Float of the angle in radians
        """
        return 2. * np.pi / self.num_vertices * spine + np.pi / 2.


    def get_xy(self, val, spine=0):
        """Returns the coordinates of a given value along a spine

        Args:
            val (float): the value along the spine
            spine (int): the index of the spine

        Returns:
            Tuple containing the coordinates as (x, y)
        """
        theta = self.get_theta(spine)
        return (float(val) * np.cos(theta) + self.radius,
                float(val) * np.sin(theta) + self.radius)


    def add_field(self, vals):
        """Adds a field to the radar plot

        Args:
            vals (list): the values along each spine for the field

        Returns:
            List containing the coordiantes of the field
        """
        assert len(vals) == self.num_vertices
        field = []
        for i, val in enumerate(vals):
            val = prepval(val, self.vertices[i])
            field.append(self.get_xy(self.radius * val / self.maxval, i))
            # Fill in the slice if that option is specified
            if self.fill:
                field.append(self.get_xy(self.radius * val / self.maxval, i + 1))
        self._fields.append(field)
        return field


    def add_slice(self, start, end=None):
        """Adds fill to a slice of the chart corresponding to start and end

        Args:
            start (mixed): the index or name of the first vertex/slice
            end (mixed): the index or name of the last vertex/slice

        Returns:
            List containing the coordiantes of the slice
        """
        if end is None:
            end  = start
        if isinstance(start, basestring):
            start = self.vertices.index(start)
        if isinstance(end, basestring):
            end = self.vertices.index(end) + 1
        field = [(self.radius, self.radius)]
        for i in xrange(start, end + 1):
            field.append(self.get_xy(self.radius, i))
        self._slices.append(field)
        return field




def build_radar_plot(data, vertices=None, title=None, caption=None,
                     fp=None, labels=None, slices=None, **kwargs):
    """Constructs a radar plot

    Args:
        data (mixed): fields to add to the plot as a list or dict
        vertices (list): the names of the fields/slices. Optional if
        title (str): the title of the plot
        caption (str): a caption for the plot
        fp (str): filepath for saving the file. If None, the plot is shown,
            not saved.
        labels (list): labels for each row in data
        slices (list): start-end pairs for slices to highlight
    """
    # Convert data to vertices and rows if given as dicts
    if vertices is None:
        vertices = data[0].keys()
    if isinstance(data[0], dict):
        data = [[row.get(v) for v in vertices] for row in data]
    _, ax = plt.subplots(1, 1, figsize=(8, 8))
    radplot = RadarPlot(vertices, **kwargs)
    ax.set_xlim(-0.05, 0.05 + radplot.radius * 2)
    ax.set_ylim(-0.05, 0.05 + radplot.radius * 2)
    if title is not None:
        title = ax.set_title(title)
    if caption is not None:
        ax.text(0.5, -0.12, caption, fontsize=9, ha='center', wrap=True)
    # Build the basic plot
    ax.add_collection(radplot.grid())
    ax.add_collection(radplot.spines())
    for x, y, label, rotation, va in radplot.labels():
        ax.text(x, y, label, rotation=rotation, fontsize=8, ha='center', va=va, wrap=True)
    # Add slices to make the plot a little more readable
    if radplot.fill and slices is not None:
        for field in slices:
            radplot.add_slice(*field)
        ax.add_collection(radplot.slices())
    # Add data
    for row in data:
        radplot.add_field(row)
    for field in radplot.fields(labels):
        ax.add_patch(field)
    if labels:
        ax.legend()
    ax.axis('off')
    # Save or show the radar plot
    if fp is not None:
        try:
            os.makedirs(os.path.dirname(fp))
        except OSError:
            pass
        print 'Saving {}...'.format(fp)
        try:
            savefig(fp, title=title)
        except IOError:
            print 'Could not write {}'.format(fp)
    else:
        plt.show()
    plt.close()



def preplabel(val):
    """Formats a label for use on the plot

    Args:
        val (str): the label

    Returns:
        Formatted label as a string
    """
    return ''.join([' ' + c if c.isupper() else c for c in val]).strip()


def prepval(val, vertex, ignore=None):
    """Prepares a value from the NMNH prioritization worksheet

    Args:
        val (float): the value for the given key
        key (str): the name of the vertex/slice
        ignore (list): list of vertices to ignore when making the correction.
            If None, defaults to the constant. If an empty list, no values are
            corrected.

    Returns:
        Corrected value as float
    """
    if ignore is None:
        ignore = IGNORE
    val = float(val)
    # The prioritization sheet for everything except Significance is set
    # up so that LOWER VALUES have HIGHER PRIORITIES. Fix values for the
    # radar plot so that the HIGHER VALUES correspond to HIGHER PRIORITIES.
    if val and ignore is not None and vertex not in ignore:
        val = 6. - val  # technically this is maxval + 1
    # Negative values are not allowed
    if val < 0:
        val = 0.
        #raise ValueError('Negative values are prohibited')
    return val


def slugify(val):
    """Converts a string to a slug appropriate for a filename or url

    Args:
        val (str): an arbitary string

    Returns:
        Slugified value as string
    """
    return re.sub(r'[^a-z0-9_\-]', '', val.replace(' ', '_').lower())



if __name__ == '__main__':
    # Quick example of how to create a plot using this scropt
    vertices = [
        'Category A',
        'Category B',
        'Category C',
        'Category D',
        'Category E',
        'Category F',
        'Category G'
        ]
    data = {
        'Dataset 1': [1., 2., 2., 4., 5., 3., 2.],
        #'Dataset 2': [4., 4., 1., 2., 3., 4., 4.],
        #'Dataset 3': [1., 1., 1., 3., 3., 1., 1.],
        #'Dataset 4': [3., 4., 4., 5., 4., 5., 1.],
    }
    datadict = {
        'Dataset 5': {
            'Category A': 1.5,
            'Category B': 2.2,
            'Category C': 3.0,
            'Category D': 3.0,
            'Category E': 0.8,
            'Category F': 1.1,
            'Category G': 4.5,
        }
    }
    for title, row in data.iteritems():
        # Plot with highlighted slices
        slices = [('Category A',), ('Category D', 'Category E')]
        build_radar_plot([row], vertices, title=title, slices=slices)
        # Plot with data marked on spine
        build_radar_plot([row], vertices, title=title, fill=False)
        # Save plot with label to a file
        fp = slugify(title) + '.png'
        build_radar_plot([row], vertices, title=title, labels=[title], fp=fp, slices=slices)
    # Plot from a dict. Note that normal dicts do not maintain order.
    for title, dct in datadict.iteritems():
        build_radar_plot([dct], title=title)
