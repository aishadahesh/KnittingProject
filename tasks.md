task 1:
i fear you applied wrong things that bugged the code. see for example the attached image, the rows are overlapping each others, which is not correct.. the duplicate per x and y should look like one big flawless model.
the glue is working good, where you detect the edges and build one big image. but the duplicate model itself is not working correctly.
another thing that is not looking good is the scanned images, if you look at the last 1 or 2 runs, you can see clearly that the model is not looking good, and the base model doesnt look as expected!
see the attached implementation plan md file that was proposed by another agent.

---------------------------------------------------------------------------------------------------------------

task 2:
Please fix the reference image behavior in **Puzzle Mode**.
When I open Puzzle Mode, the reference image should not be displayed in the background.
I noticed this issue happens especially after clicking **Reset Initial Model**. After resetting, the initial model appears with the reference image background, even when I am in Puzzle Mode. This is incorrect.
Expected behavior:
1. The reference image should only be available in **Edit Mode**.
2. The reference image should never appear in **Puzzle Mode**.
3. The **Reset Initial Model** button should only be clickable/available in **Edit Mode**.
4. If I switch from Edit Mode to Puzzle Mode while the reference image checkbox is enabled, the reference image should be hidden automatically.
5. If I return back to Edit Mode, the reference image should appear again only if it was enabled before.
6. Switching between Edit Mode and Puzzle Mode should not permanently lose the reference image checkbox state.
7. Puzzle Mode should stay clean and show only puzzle-related content.
In short: hide the reference image in Puzzle Mode, make **Reset Initial Model** available only in Edit Mode, and restore the reference image state when returning to Edit Mode.


---------------------------------------------------------------------------------------------------------------

task 3:
In **Scan Mode**, add a duplicate-scan prevention step.
Before starting a new scan, check whether the same scan configuration already exists in the database.
A scan should be considered already existing if the important parameters match, such as:
* selected colors,
* generated patterns / bitmaps,
* lighting mode,
* camera angles,
* image repeat X/Y,
* scan settings,
* capture mode,
* any other parameters that affect the final scanned images or RGB analysis.

Expected behavior:
1. When the user clicks **Start Scanner**, first compare the current scan configuration with existing database entries.
2. If an identical configuration already exists, do not immediately repeat the scan.
3. Display a clear message:

```text
This scan already exists in the database.
Do you want to rescan it or cancel?
```

4. Give the user two options:
   * **Rescan**: run the scanning process again and save it as a new scan run.
   * **Cancel**: stop the scan and keep the existing database result.
5. If the user chooses **Cancel**, no new images or metadata should be generated.
6. If the user chooses **Rescan**, create a new scan run ID so the new result does not overwrite the old one.
7. The duplicate check should prevent unnecessary repeated scanning of the same scene and save time.

In short: before scanning, check if the same configuration is already stored in the database. If it exists, ask the user whether to rescan or cancel instead of automatically repeating the same scan.




-------------
in the puzzle mode: 
if i switch to the edit mode then to puzzle mode, so the preview in the puzzle mode is the same model editd in the edit mode, but if i go to the scan mode then switch to the puzzle mode, so the preview is of the scan mode. this behaviour is not correct, since the puzzle mode should be independent, and have its own pattern editing. 

----------

- in ur5 real  robot motion mode, a camera should be connected to robot. detect the sign of calibration to know where everything is located, then for the fabric scan- we should be in the center of fabric, and sca from multiple views, while making sure trigger is not getting too high, to make sure photos are actully contaning meaningful data.


- enhance the simulator and gui (app) . let it be much more close to the env we achieve for

- teach mode just for the z axes, so robot can learn the wanted zoom..